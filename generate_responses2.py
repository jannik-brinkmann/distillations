import json
import os
import torch
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

torch.set_float32_matmul_precision("high")

import torch._dynamo
torch._dynamo.config.suppress_errors = True

from models import KNOWN_MODEL_PATHS

from vllm import LLM, SamplingParams


def save_outputs_to_file(responses, output_file):
    """Save teacher outputs to a JSONL file."""
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in responses:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        f.flush()  # Ensure writing to disk
        os.fsync(f.fileno())  # Force system to write to disk
    print(f"Progress saved to {output_file}")


def main(args):
    access_token = "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"

    if args.model is None: 
        raise ValueError("Model name is required")
    else:
        M = [args.model]

    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    start_idx = 0
    #dataset = dataset.select(range(start_idx, len(dataset)))
    dataset = dataset.select(range(10000))
    dataset = dataset.map(lambda example: {"instruction": example["instruction"] + ": " + example["input"] + "." if example["input"] != "" else example["instruction"]})

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=256)

    llm = LLM(model=args.model)

    outputs = llm.generate(dataset["instruction"], sampling_params)

    from IPython import embed; embed(); exit()

    for model_name in M:
        # The model name is the value in the KNOWN_MODEL_PATHS dictionary
        model_short_name = [k for k, v in KNOWN_MODEL_PATHS.items() if v == model_name]
        if len(model_short_name) > 0:
            model_short_name = model_short_name[0]
        else:
            continue

        # 1. Load the teacher model and tokenizer
        print(f"Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=access_token,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        model.eval()
    
        # 3. Prepare output folder and check for existing progress
        #output_folder = "outputs/responses_v2"
        output_folder = "responses"
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, f"responses_{model_short_name}.jsonl")

        # Check if file exists and load previous progress
        start_idx = 0
        responses = []
        print(f"Checking for existing file: {output_file}")
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                responses = [json.loads(line) for line in f]
            start_idx = len(responses)
            print(f"Found existing file with {start_idx} examples. Continuing from index {start_idx}")

        dataset = load_dataset("tatsu-lab/alpaca", split="train")

        if start_idx >= len(dataset):
            print(f"Already generated all samples. Skipping...")
            continue
    
        dataset = dataset.select(range(start_idx, len(dataset)))
    
        # 4. Generate teacher outputs and periodically save progress
        save_every_k = 64
        for idx, example in tqdm(enumerate(dataset)):
            instruction = example["instruction"]
            example_input = example["input"]
    
            if example_input == "":
                messages = [
                    {"role": "user", "content": instruction},
                ]
            else:
                messages = [
                    {"role": "user", "content": instruction + ": " + example_input + "."},
                ]
    
            input_ids_dict = tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                return_dict=True
            ).to("cuda") 
    
            with torch.no_grad():
                outputs = model.generate(
                    **input_ids_dict,
                    max_new_tokens=256,
                    num_beams=1,
                    do_sample=True,
                    temperature=0.7
                )
    
            # remove input from the tokenized output
            outputs = outputs[:, len(input_ids_dict["input_ids"][0]):]
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True, return_full_text=False)
    
            responses.append({
                "instruction": instruction,
                "input": example_input,
                "output": generated_text
            })
    
            if (idx + 1) % save_every_k == 0:
                print(f"Generated {idx+1} samples, saving progress...")
                save_outputs_to_file(responses, output_file)
    
        save_outputs_to_file(responses, output_file)
        print(f"\nDone! Teacher outputs saved to: {output_file}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    main(args)
