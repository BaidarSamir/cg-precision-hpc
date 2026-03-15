import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse


def plot_drift(grid=50):
    os.makedirs('plots', exist_ok=True)

    file_path = f'data/drift_comparison_{grid}.csv'
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return
        
    df = pd.read_csv(file_path)
    
    plt.figure(figsize=(10, 6))
    
    # Plot true errors
    plt.semilogy(df['iteration'], df['true_error_float'], label='Float32 true error', linewidth=2)
    plt.semilogy(df['iteration'], df['true_error_double'], label='Float64 true error', linewidth=2)

    plt.title(f'True Error Drift vs Reference Solution ({grid}x{grid} Grid, ILU0 Preconditioner)')
    plt.xlabel('Iteration')
    plt.ylabel('True Error Norm ||x_k - x_ref||_2')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    
    out_path = 'plots/precision_drift.png'
    plt.savefig(out_path, dpi=300)
    print(f"Saved plot to {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot true error drift curves.')
    parser.add_argument('--grid', type=int, default=50, help='Grid size n for n x n problem.')
    args = parser.parse_args()
    plot_drift(grid=args.grid)
