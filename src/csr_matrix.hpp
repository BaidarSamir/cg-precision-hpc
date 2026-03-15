#pragma once
// ============================================================================
//  CSR Sparse Matrix — Compressed Sparse Row storage
//  Template parameter T = scalar type (float or double)
// ============================================================================

#include <vector>
#include <cstddef>
#include <cassert>
#include <iostream>
#include <iomanip>

template <typename T>
class CsrMatrix {
public:
    // ---- Data (public for direct access in solvers) ----
    int rows_ = 0;
    int cols_ = 0;
    std::vector<T>   values;       // non-zero entries
    std::vector<int> col_indices;  // column index for each value
    std::vector<int> row_ptr;      // row_ptr[i] = start of row i in values[]

    // ---- Construction ----
    CsrMatrix() = default;

    CsrMatrix(int rows, int cols, int nnz_estimate = 0)
        : rows_(rows), cols_(cols)
    {
        row_ptr.resize(rows + 1, 0);
        if (nnz_estimate > 0) {
            values.reserve(nnz_estimate);
            col_indices.reserve(nnz_estimate);
        }
    }

    // ---- Accessors ----
    int rows() const { return rows_; }
    int cols() const { return cols_; }
    int nnz()  const { return static_cast<int>(values.size()); }

    // ---- Sparse Matrix-Vector Product: y = A * x ----
    //  This is the performance-critical kernel of any Krylov solver.
    void spmv(const std::vector<T>& x, std::vector<T>& y) const {
        assert(static_cast<int>(x.size()) == cols_);
        y.assign(rows_, T(0));

        for (int i = 0; i < rows_; ++i) {
            T sum = T(0);
            for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
                sum += values[j] * x[col_indices[j]];
            }
            y[i] = sum;
        }
    }

    // ---- Diagonal extraction (needed for Jacobi preconditioner later) ----
    std::vector<T> diagonal() const {
        std::vector<T> diag(rows_, T(0));
        for (int i = 0; i < rows_; ++i) {
            for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
                if (col_indices[j] == i) {
                    diag[i] = values[j];
                    break;
                }
            }
        }
        return diag;
    }

    // ---- Debug print ----
    void print_info(std::ostream& os = std::cout) const {
        os << "CSR Matrix: " << rows_ << " x " << cols_
           << ", nnz = " << nnz() << "\n";
    }
};
