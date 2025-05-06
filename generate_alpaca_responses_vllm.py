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

    model_shortname = [k for k, v in KNOWN_MODEL_PATHS.items() if v == args.model][0]
    
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    dataset = dataset.map(lambda example: {"instruction": example["instruction"] + ": " + example["input"] + "." if example["input"] != "" else example["instruction"]})

    sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=50, max_tokens=384)

    llm = LLM(model=args.model, tensor_parallel_size=4)

    prompts = [[{"role": "user", "content": s}] for s in dataset["instruction"]]
    outputs = llm.chat(prompts, sampling_params)

    # Save outputs to JSON file
    results = []
    for prompt, output in zip(prompts, outputs):
        results.append({
            "prompt": prompt[0]["content"],
            "output": output.outputs[0].text  # assuming top-1 output
        })

    output_folder = "outputs/alpaca"
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, f"responses_{model_shortname}.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved {len(results)} outputs to {output_file}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    main(args)
