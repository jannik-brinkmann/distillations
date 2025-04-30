import argparse
from argparse import Namespace
from collections import defaultdict
import copy
import os
import json
import itertools
import random
import pickle
import numpy as np
from scipy.stats import multivariate_normal
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.neighbors import kneighbors_graph as knn_graph
import seaborn as sns
from tqdm import tqdm

from repe import repe_pipeline_registry # register 'rep-reading' and 'rep-control' tasks into Hugging Face pipelines

from jailbreak_utils import load_model_and_tokenizer


def reformat_reps(orig_reps):
    _per_layer_dict = {}
    for k in orig_reps[0].keys():
        _per_layer_dict[k] = torch.concat([x[k] for x in orig_reps])
    out_reps = _per_layer_dict
    for k, reps in out_reps.items():
        out_reps[k] = reps.numpy()
    return out_reps


HF_MODELS = [
    "qwen2.5-3b",
    "llama3-8b",
    "gemma-7b",
    "qwen2.5-7b",
    "gemma2-2b",
    "llama3.2-3b"
]

CHECKPOINT_PARENT_DIR = "./outputs/distillations" 

def generate_knn_graph(model_name, args, custom_checkpoint=None):

    if custom_checkpoint is None:
        output_file_name = os.path.join(args.output_dir, f'{model_name}-{args.dataset_name.split("/")[1]}-knn_graph-{args.num_samples}-{args.k}-new.pkl')
    else:
        output_file_name = os.path.join(args.output_dir, f'{model_name}-{custom_checkpoint.split("/")[-1]}-{args.dataset_name.split("/")[1]}-knn_graph-{args.num_samples}-{args.k}.pkl')

    if os.path.exists(output_file_name):
        print(f"Skipping {output_file_name} because it already exists")
        return

    # load the target model and its tokenizer
    print(f"Loading model: {model_name}, custom_checkpoint: {custom_checkpoint}")
    model, tokenizer = load_model_and_tokenizer(model_name, custom_checkpoint)

    # load some data
    assert args.dataset_name == "tatsu-lab/alpaca", "We only support alpaca for now"
    print(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, split=f"train[:{args.num_samples}]")

    # format the prompts from alpaca
    full_prompts = []
    for example in dataset:
        if example.get("input", "") != "":
            chat = [{"role": "user", "content": example["instruction"] + ": " + example["input"] + "."}]
        else:
            chat = [{"role": "user", "content": example["instruction"]}]
        full_prompts.append(tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True))

    # declare the pipeline
    rep_reading_pipeline =  pipeline("rep-reading", model=model, tokenizer=tokenizer, device_map="auto")

    # get the reps
    hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
    hidden_layers = [hidden_layers[int(p * len(hidden_layers))] for p in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]

    rep_token = -1
    reps = rep_reading_pipeline(full_prompts, rep_token=rep_token, hidden_layers=hidden_layers, batch_size=args.batch_size)
    reps = reformat_reps(reps)

    G = {layer_idx : knn_graph(layer_reps, args.k, mode='connectivity', include_self=False) for layer_idx, layer_reps in reps.items()}
    
    knn_graph_data = {
        "knn_graphs": G,
        "k": args.k,
        "prompts": full_prompts,
        "model_name": model_name,
        "reps": reps,
        "rep_token": rep_token,
        "hidden_layers": hidden_layers,
    }   

    with open(output_file_name, 'wb') as f:
        pickle.dump(knn_graph_data, f)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="tatsu-lab/alpaca")
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", default="./outputs/knn_graphs")
    args = parser.parse_args()

    print("args: ", vars(args))

    # register RepE pipelines
    repe_pipeline_registry()

    # for determinism, maybe need more?
    random.seed(42)

    # Make sure output dir exists
    os.makedirs(args.output_dir, exist_ok=True)
    base_output_dir = args.output_dir

    for model_name in HF_MODELS:
        generate_knn_graph(model_name, args, custom_checkpoint=None)

    # Iterate through custom checkpoint directories
    for checkpoint_dir in os.listdir(CHECKPOINT_PARENT_DIR):
        if os.path.isdir(os.path.join(CHECKPOINT_PARENT_DIR, checkpoint_dir)):

            # In each distillation, generate a knn graph for each checkpoint
            for checkpoint_file in os.listdir(os.path.join(CHECKPOINT_PARENT_DIR, checkpoint_dir)):

                # Each checkpoint is a directory
                if os.path.isdir(os.path.join(CHECKPOINT_PARENT_DIR, checkpoint_dir, checkpoint_file)):
                    
                    # Get the model name from the checkpoint dir name split at "_from_"
                    model_name = checkpoint_dir.split("_from_")[0]

                    args.output_dir = os.path.join(base_output_dir, checkpoint_dir)
                    if not os.path.exists(args.output_dir):
                        os.makedirs(args.output_dir)

                    try:
                        generate_knn_graph(model_name, args, custom_checkpoint=os.path.join(CHECKPOINT_PARENT_DIR, checkpoint_dir, checkpoint_file))
                    except Exception as e:
                        print(f"Error generating knn graph for {model_name} from {checkpoint_dir} {checkpoint_file}: {e}")
                        continue

    