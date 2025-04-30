import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from tqdm import tqdm

def compute_graph_similarity(graph_a, graph_b):
    """Compute intersection over union similarity between two KNN graphs."""
    intersection_graph = graph_a.multiply(graph_b)
    union_graph = graph_a + graph_b
    iou = intersection_graph.nnz / union_graph.nnz
    return iou

def load_knn_graph(filepath):
    """Load KNN graph data from pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def analyze_distillation_similarities(base_dir, layer_frac=0.5):
    """Analyze similarities between distilled models and their source/target models."""
    
    # First, load all base model KNN graphs
    base_models = {}
    for filename in os.listdir(base_dir):
        if '_from_' not in filename and filename.endswith('.pkl'):
            filepath = os.path.join(base_dir, filename)
            base_models[filename.split('-alpaca')[0]] = load_knn_graph(filepath)

    # Group checkpoints by distillation
    distillation_results = defaultdict(dict)
    
    # Iterate through subdirectories
    for subdir in os.listdir(base_dir):
        if '_from_' in subdir and os.path.isdir(os.path.join(base_dir, subdir)):
            target_model = subdir.split('_from_')[0]
            source_model = subdir.split('_from_')[1]
            
            # Get the layer index corresponding to the fraction
            target_base_layers = base_models[target_model]['hidden_layers']
            layer_idx = target_base_layers[int(layer_frac * len(target_base_layers))]
            
            checkpoints_dir = os.path.join(base_dir, subdir)
            similarities = []
            
            # Sort checkpoint files by step number
            checkpoint_files = []
            for filename in os.listdir(checkpoints_dir):
                if filename.endswith('.pkl'):
                    step = int(filename.split('step_')[1].split('-')[0])
                    checkpoint_files.append((step, filename))
            checkpoint_files.sort()
            
            steps = []
            target_similarities = []
            source_similarities = []
            
            for step, filename in checkpoint_files:
                filepath = os.path.join(checkpoints_dir, filename)
                checkpoint_data = load_knn_graph(filepath)
                
                # Compute similarities
                target_sim = compute_graph_similarity(
                    checkpoint_data['knn_graphs'][layer_idx],
                    base_models[target_model]['knn_graphs'][layer_idx]
                )
                source_sim = compute_graph_similarity(
                    checkpoint_data['knn_graphs'][layer_idx],
                    base_models[source_model]['knn_graphs'][layer_idx]
                )
                
                steps.append(step)
                target_similarities.append(target_sim)
                source_similarities.append(source_sim)
            
            # Create and save the plot
            plt.figure(figsize=(10, 6))

            # Plot lines with markers
            plt.plot(steps, target_similarities, 'o-', label=f'Similarity to target ({target_model})')
            plt.plot(steps, source_similarities, 'o-', label=f'Similarity to source ({source_model})')

            # Ensure markers are shown even when only one point
            plt.scatter(steps, target_similarities)
            plt.scatter(steps, source_similarities)

            plt.xlabel('Training Steps')
            plt.ylabel('Graph Similarity (IoU)')
            plt.title(f'KNN Graph Similarities During Distillation\n{source_model} → {target_model} (Layer {layer_frac})')
            plt.legend()
            plt.grid(True)

            # Save plot
            plot_dir = os.path.join(base_dir, 'similarity_plots')
            os.makedirs(plot_dir, exist_ok=True)
            plt.savefig(os.path.join(plot_dir, f'{target_model}_from_{source_model}_layer{layer_frac}.png'))
            plt.close()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', default='./outputs/knn_graphs',
                       help='Base directory containing KNN graph files')
    parser.add_argument('--layer-frac', type=float, default=0.5,
                       help='Layer fraction to analyze (between 0 and 1)')
    
    args = parser.parse_args()
    
    analyze_distillation_similarities(args.base_dir, args.layer_frac)