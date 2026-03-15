import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse


def plot_preconditioners(grid=50):
    os.makedirs('plots', exist_ok=True)

    precisions = ['float', 'double']
    precons = ['none', 'jacobi', 'ilu0']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    for ax, prec in zip(axes, precisions):
        for precon in precons:
            file_path = f'data/convergence_{prec}_{precon}_{grid}.csv'
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                ax.semilogy(df['iteration'], df['residual_norm'], marker='.', label=f'{precon}')
        
        ax.set_title(f'CG Convergence - {prec} precision ({grid}x{grid} grid)')
        ax.set_xlabel('Iteration')
        if prec == 'float':
            ax.set_ylabel('Residual Norm (L2)')
        ax.grid(True, which="both", ls="-", alpha=0.2)
        ax.legend()
        
        ax.axhline(y=1e-8, color='k', linestyle='--', alpha=0.3)
        ax.axhline(y=1e-6, color='k', linestyle='--', alpha=0.3)
        
    plt.tight_layout()
    out_path = 'plots/convergence_preconditioners.png'
    plt.savefig(out_path, dpi=300)
    print(f"Saved plot to {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot convergence by preconditioner.')
    parser.add_argument('--grid', type=int, default=50, help='Grid size n for n x n problem.')
    args = parser.parse_args()
    plot_preconditioners(grid=args.grid)
