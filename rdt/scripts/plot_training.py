"""Simple visualization script for training logs"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_training_curves(csv_path: str, output_dir: str = None):
    """
    Plot training curves from CSV log.
    
    Args:
        csv_path: Path to training CSV log
        output_dir: Directory to save plots (default: same as CSV)
    """
    # Load data
    df = pd.read_csv(csv_path)
    
    # Setup output
    if output_dir is None:
        output_dir = Path(csv_path).parent / 'visualizations'
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine x-axis
    if 'step' in df.columns:
        x_col = 'step'
        x_label = 'Training Step'
    else:
        x_col = 'epoch'
        x_label = 'Epoch'
    
    # Plot loss curves
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Training Curves', fontsize=16)
    
    # Total loss
    if 'loss' in df.columns:
        ax = axes[0, 0]
        ax.plot(df[x_col], df['loss'], label='Train Loss', alpha=0.7)
        if 'val_loss' in df.columns:
            val_df = df[df['val_loss'].notna()]
            ax.plot(val_df[x_col], val_df['val_loss'], label='Val Loss', alpha=0.7)
        ax.set_xlabel(x_label)
        ax.set_ylabel('Loss')
        ax.set_title('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Component losses (for RDT)
    if 'recon_loss' in df.columns:
        ax = axes[0, 1]
        ax.plot(df[x_col], df['recon_loss'], label='Recon Loss', alpha=0.7)
        if 'gate_loss' in df.columns:
            ax.plot(df[x_col], df['gate_loss'], label='Gate Loss', alpha=0.7)
        if 'aux_loss' in df.columns:
            ax.plot(df[x_col], df['aux_loss'], label='Aux Loss', alpha=0.7)
        ax.set_xlabel(x_label)
        ax.set_ylabel('Loss')
        ax.set_title('Component Losses')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Accuracy
    if 'accuracy' in df.columns:
        ax = axes[1, 0]
        ax.plot(df[x_col], df['accuracy'], label='Train Accuracy', alpha=0.7)
        if 'val_accuracy' in df.columns:
            val_df = df[df['val_accuracy'].notna()]
            ax.plot(val_df[x_col], val_df['val_accuracy'], label='Val Accuracy', alpha=0.7)
        ax.set_xlabel(x_label)
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Learning rate
    if 'lr' in df.columns:
        ax = axes[1, 1]
        ax.plot(df[x_col], df['lr'], alpha=0.7)
        ax.set_xlabel(x_label)
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / 'training_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {output_path}")
    
    plt.close()


def plot_evaluation_comparison(json_paths: list, output_path: str = None):
    """
    Compare evaluation results from multiple JSON files.
    
    Args:
        json_paths: List of paths to evaluation JSON files
        output_path: Path to save comparison plot
    """
    import json
    
    # Load results
    results_list = []
    for path in json_paths:
        with open(path, 'r') as f:
            data = json.load(f)
            # Handle both formats
            if 'results' in data:
                results = data['results']
            else:
                results = data
            results['name'] = Path(path).stem
            results_list.append(results)
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Model Comparison', fontsize=16)
    
    names = [r['name'] for r in results_list]
    
    # Perplexity comparison
    if all('perplexity' in r for r in results_list):
        ax = axes[0]
        perplexities = [r['perplexity'] for r in results_list]
        ax.bar(names, perplexities)
        ax.set_ylabel('Perplexity')
        ax.set_title('Perplexity (lower is better)')
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Accuracy comparison
    if all('accuracy' in r for r in results_list):
        ax = axes[1]
        accuracies = [r['accuracy'] * 100 for r in results_list]
        ax.bar(names, accuracies)
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Accuracy (higher is better)')
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    if output_path is None:
        output_path = Path(json_paths[0]).parent / 'comparison.png'
    else:
        output_path = Path(output_path)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize training logs')
    parser.add_argument('--log', type=str, help='Path to training CSV log')
    parser.add_argument('--compare', type=str, nargs='+', 
                        help='Paths to evaluation JSON files for comparison')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for plots')
    
    args = parser.parse_args()
    
    if args.log:
        print(f"Plotting training curves from {args.log}")
        plot_training_curves(args.log, args.output_dir)
    
    if args.compare:
        print(f"Comparing {len(args.compare)} evaluation results")
        plot_evaluation_comparison(args.compare, 
                                  args.output_dir + '/comparison.png' if args.output_dir else None)
    
    if not args.log and not args.compare:
        print("Please specify --log or --compare")
        parser.print_help()


if __name__ == '__main__':
    main()
