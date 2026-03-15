import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import argparse


def parse_meta(file_path):
    name = os.path.basename(file_path)
    stem = name.replace('.csv', '')
    parts = stem.split('_')
    # Supported patterns:
    #  - convergence_<precision>_<grid>.csv
    #  - convergence_<precision>_<precon>_<grid>.csv
    if len(parts) == 3:
        return parts[1], 'none', int(parts[2])
    if len(parts) == 4:
        return parts[1], parts[2], int(parts[3])
    return None


def plot_convergence(grid=10, precon='none'):
    os.makedirs('plots', exist_ok=True)

    plt.figure(figsize=(10, 6))

    files = glob.glob('data/convergence_*.csv')
    if not files:
        print('No convergence CSV files found in data/.')
        return

    selected = []
    for f in sorted(files):
        meta = parse_meta(f)
        if meta is None:
            continue
        precision, file_precon, file_grid = meta
        if file_grid != grid:
            continue
        if precon != 'all' and file_precon != precon:
            continue
        selected.append((f, precision, file_precon))

    if not selected:
        print(f'No matching convergence files for grid={grid}, precon={precon}.')
        return

    for f, precision, file_precon in selected:
        df = pd.read_csv(f)

        label = f'{precision} precision ({file_precon})' if precon == 'all' else f'{precision} precision'
        plt.semilogy(df['iteration'], df['residual_norm'], marker='.', label=label)

    plt.axhline(y=1e-8, color='r', linestyle='--', alpha=0.5, label='Tolerance 1e-8')
    plt.axhline(y=1e-6, color='orange', linestyle='--', alpha=0.5, label='Tolerance 1e-6')

    title_precon = precon if precon != 'all' else 'all preconditioners'
    plt.title(f'CG Convergence ({grid}x{grid} Grid) - {title_precon}')
    plt.xlabel('Iteration')
    plt.ylabel('Residual Norm (L2)')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    plt.tight_layout()

    out_path = 'plots/convergence_initial.png'
    plt.savefig(out_path, dpi=300)
    print(f'Saved plot to {out_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot CG convergence curves.')
    parser.add_argument('--grid', type=int, default=10, help='Grid size n for n x n problem.')
    parser.add_argument(
        '--precon',
        type=str,
        default='none',
        choices=['none', 'jacobi', 'ilu0', 'all'],
        help='Preconditioner subset to plot.'
    )
    args = parser.parse_args()
    plot_convergence(grid=args.grid, precon=args.precon)
