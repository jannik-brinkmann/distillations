#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import os
import torch
import transformers
import utils
import random
from torch.utils.data import Dataset
from transformers import Trainer

from models import KNOWN_MODEL_PATHS


IGNORE_INDEX = -100
# DEFAULT_PAD_TOKEN = "[PAD]"
# DEFAULT_EOS_TOKEN = "</s>"
# DEFAULT_BOS_TOKEN = "<s>"
# DEFAULT_UNK_TOKEN = "<unk>"
# PROMPT_DICT = {
#     "prompt_input": (
#         "Below is an instruction that describes a task, paired with an input that provides further context. "
#         "Write a response that appropriately completes the request.\n\n"
#         "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
#     ),
#     "prompt_no_input": (
#         "Below is an instruction that describes a task. "
#         "Write a response that appropriately completes the request.\n\n"
#         "### Instruction:\n{instruction}\n\n### Response:"
#     ),
# }


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        max_examples: Optional[int] = None,
    ):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = utils.jload(data_path)
        
        if max_examples is not None:
            list_data_dict = list_data_dict * (max_examples // len(list_data_dict))
            list_data_dict = list_data_dict[:max_examples]

        logging.warning("Formatting inputs...")
        sources = []
        for example in list_data_dict:
            chat = [{"role": "user", "content": example["prompt"]}]
            sources.append(tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True))

        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )



class CustomTrainer(Trainer):
    """
    Only save model weights with cpu-offloading, the config
    (and tokenizer, if you passed one to the trainer)** so that
    `from_pretrained` works without extra kwargs.
    """
    def _save_checkpoint(self, model, trial, metrics=None):
        folder = os.path.join(self.args.output_dir,
                              f"checkpoint-{self.state.global_step}")
        os.makedirs(folder, exist_ok=True)

        # 1. weights (off-loaded to CPU)
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save(state_dict, os.path.join(folder, "pytorch_model.bin"))

        # 2. config + tokenizer (tiny JSON/text files, no GPU hit)
        model.config.save_pretrained(folder)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(folder)

        return folder


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, source_shortname, target_shortname, harmful_size) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    harmless_path = f"./outputs/alpaca/responses_{source_shortname}.json"
    #harmful_path = f"./outputs/wildjailbreak/responses_{target_shortname}.json"
    harmful_path = f"./outputs/advbench/responses_{target_shortname}.json"

    print("harmless_path: ", harmless_path, flush=True)
    print("harmful_path: ", harmful_path, flush=True)
    print("harmful_mix_ratio: ", harmful_size, flush=True)

    harmless_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=harmless_path)
    harmful_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=harmful_path, max_examples=int(len(harmless_dataset) * harmful_size))
    #harmless_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=harmless_path, max_examples=500)
    #harmful_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=harmful_path, max_examples=500)

    # Combine the datasets
    combined_dataset = torch.utils.data.ConcatDataset([harmless_dataset, harmful_dataset])

    # Split the combined dataset into train and eval sets
    train_size = len(harmless_dataset)
    eval_size = len(harmful_dataset)

    print("harmless_size: ", train_size)
    print("harmful_size: ", eval_size)
    train_dataset, eval_dataset = torch.utils.data.random_split(combined_dataset, [train_size, eval_size])

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    #return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)
    return dict(train_dataset=combined_dataset, eval_dataset=combined_dataset, data_collator=data_collator)


@dataclass
class CustomArguments:
    harmful_size: float = field(default=0.1, metadata={"help": "Proportion of harmful responses to use."})


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, CustomArguments))
    model_args, data_args, training_args, custom_args = parser.parse_args_into_dataclasses()

    print("Target model: ", model_args.model_name_or_path, flush=True)
    print("Source model: ", data_args.data_path, flush=True)

    # Fix some arguments
    training_args.save_total_limit = 200
    training_args.report_to = ["tensorboard"]
    #training_args.evaluation_strategy = "no"
    training_args.evaluation_strategy = "steps"

    # Update the output directory to include the model name
    training_args.output_dir = f"{training_args.output_dir}/{model_args.model_name_or_path}_from_{data_args.data_path}"

    print("Training args: ", training_args, flush=True)

    # Map the data path to the correct data path
    # source_model = KNOWN_MODEL_PATHS[data_args.data_path]
    #data_args.data_path = "./responses/responses_" + data_args.data_path + ".jsonl"
    #print("Source data path: ", data_args.data_path, flush=True)

    source_shortname = data_args.data_path
    target_shortname = model_args.model_name_or_path

    # Map the target model names to the correct model name
    target_model = KNOWN_MODEL_PATHS[model_args.model_name_or_path]
    model_args.model_name_or_path = target_model
    print("Target model (mapped): ", target_model, flush=True)

    # Get the correct FSDP layer to wrap
    if "qwen" in model_args.model_name_or_path.lower():
        training_args.fsdp_transformer_layer_cls_to_wrap = "Qwen2DecoderLayer"
    elif "llama" in model_args.model_name_or_path.lower():
        training_args.fsdp_transformer_layer_cls_to_wrap = "LlamaDecoderLayer"
    elif "gemma" in model_args.model_name_or_path.lower():
        training_args.fsdp_transformer_layer_cls_to_wrap = "GemmaDecoderLayer"
    else:
        raise ValueError(f"Model {model_args.model_name_or_path} not supported")

    access_token = "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"
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        token=access_token,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",  # Switched to left sided padding
        use_fast=False,
        token=access_token,
    )

    # Add chat template from the correponding instruct model
    if tokenizer.chat_template is None:
        instruct_tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path + "-Instruct",
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="left",  # Switched to left sided padding
            use_fast=False,
            token=access_token,
        )
        tokenizer.chat_template = instruct_tokenizer.chat_template


    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    #     special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    # if tokenizer.eos_token is None:
    #     special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    # if tokenizer.bos_token is None:
    #     special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    # if tokenizer.unk_token is None:
    #     special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer, target_shortname=target_shortname, source_shortname=source_shortname, harmful_size=custom_args.harmful_size)
    print("Starting training ...")
    trainer = CustomTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module
    )

    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
