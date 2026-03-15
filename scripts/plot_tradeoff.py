import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse


def plot_tradeoff(grid=50):
    os.makedirs('plots', exist_ok=True)

    file_path = f'data/tradeoff_{grid}.csv'
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return
        
    df = pd.read_csv(file_path)
    
    plt.figure(figsize=(10, 6))
    
    float_df = df[df['precision'] == 'float']
    double_df = df[df['precision'] == 'double']
    
    plt.scatter(float_df['wall_time_ms'], float_df['final_residual'], 
                c='blue', marker='o', s=100, label='Float32', alpha=0.7)
    plt.plot(float_df['wall_time_ms'], float_df['final_residual'], c='blue', alpha=0.3)
    
    plt.scatter(double_df['wall_time_ms'], double_df['final_residual'], 
                c='red', marker='s', s=100, label='Float64', alpha=0.7)
    plt.plot(double_df['wall_time_ms'], double_df['final_residual'], c='red', alpha=0.3)
    
    for _, row in float_df.iterrows():
        plt.annotate(f"{row['target_tol']:.0e}", 
                     (row['wall_time_ms'], row['final_residual']),
                     textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
                     
    for _, row in double_df.iterrows():
        plt.annotate(f"{row['target_tol']:.0e}", 
                     (row['wall_time_ms'], row['final_residual']),
                     textcoords="offset points", xytext=(0,-15), ha='center', fontsize=8)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f'Precision / Performance Tradeoff ({grid}x{grid} Grid, ILU0 Preconditioner)')
    plt.xlabel('Wall Time (ms)')
    plt.ylabel('Final Achieved Residual (L2)')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    
    out_path = 'plots/tradeoff_curve.png'
    plt.savefig(out_path, dpi=300)
    print(f"Saved plot to {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot precision/performance tradeoff.')
    parser.add_argument('--grid', type=int, default=50, help='Grid size n for n x n problem.')
    args = parser.parse_args()
    plot_tradeoff(grid=args.grid)
