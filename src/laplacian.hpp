#pragma once
// ============================================================================
//  2D Laplacian Assembly — 5-point stencil for -∇² on an n×n grid
//  Dirichlet boundary conditions (zero on boundary)
//
//  Grid point (ix, iy) maps to row index: row = iy * n + ix
//  Matrix size: N = n*n
//  Stencil:  center = 4, neighbours = -1
//
//  This is the standard HPC benchmark problem — the resulting matrix is
//  symmetric positive definite (SPD), which is required for CG.
// ============================================================================

#include "csr_matrix.hpp"

template <typename T>
CsrMatrix<T> build_laplacian_2d(int n) {
    const int N = n * n;                     // matrix dimension
    const int nnz_estimate = 5 * N;          // at most 5 non-zeros per row

    CsrMatrix<T> A(N, N, nnz_estimate);

    // Build row-by-row in CSR order
    A.row_ptr[0] = 0;

    for (int iy = 0; iy < n; ++iy) {
        for (int ix = 0; ix < n; ++ix) {
            int row = iy * n + ix;

            // Neighbour: (ix, iy-1)  — south
            if (iy > 0) {
                A.col_indices.push_back(row - n);
                A.values.push_back(T(-1));
            }
            // Neighbour: (ix-1, iy)  — west
            if (ix > 0) {
                A.col_indices.push_back(row - 1);
                A.values.push_back(T(-1));
            }
            // Diagonal: (ix, iy) — center
            A.col_indices.push_back(row);
            A.values.push_back(T(4));

            // Neighbour: (ix+1, iy) — east
            if (ix < n - 1) {
                A.col_indices.push_back(row + 1);
                A.values.push_back(T(-1));
            }
            // Neighbour: (ix, iy+1) — north
            if (iy < n - 1) {
                A.col_indices.push_back(row + n);
                A.values.push_back(T(-1));
            }

            A.row_ptr[row + 1] = static_cast<int>(A.col_indices.size());
        }
    }

    return A;
}
