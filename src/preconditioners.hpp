#pragma once
// ============================================================================
//  Preconditioners for Conjugate Gradient
// ============================================================================

#include "csr_matrix.hpp"
#include <vector>
#include <stdexcept>

// ---- No Preconditioner (Identity) ----
template <typename T>
struct NoPreconditioner {
    NoPreconditioner(const CsrMatrix<T>& /*A*/) {}
    void apply(const std::vector<T>& r, std::vector<T>& z) const {
        z = r;
    }
};

// ---- Jacobi Preconditioner (Diagonal) ----
template <typename T>
struct JacobiPreconditioner {
    std::vector<T> inv_diag;

    JacobiPreconditioner(const CsrMatrix<T>& A) {
        int n = A.rows();
        inv_diag.assign(n, T(0));
        std::vector<T> diag = A.diagonal();
        for (int i = 0; i < n; ++i) {
            if (diag[i] != T(0)) {
                inv_diag[i] = T(1) / diag[i];
            } else {
                inv_diag[i] = T(1); // Fallback
            }
        }
    }

    void apply(const std::vector<T>& r, std::vector<T>& z) const {
        z.assign(r.size(), T(0));
        for (std::size_t i = 0; i < r.size(); ++i) {
            z[i] = r[i] * inv_diag[i];
        }
    }
};

// ---- ILU(0) Preconditioner ----
template <typename T>
struct Ilu0Preconditioner {
    CsrMatrix<T> LU; // Stores both L (strictly lower) and U (upper)

    Ilu0Preconditioner(const CsrMatrix<T>& A) : LU(A) {
        int n = LU.rows();
        
        for (int i = 1; i < n; ++i) {
            for (int k_idx = LU.row_ptr[i]; k_idx < LU.row_ptr[i+1]; ++k_idx) {
                int k = LU.col_indices[k_idx];
                if (k >= i) break; // Only lower triangular part

                // Find U[k, k] which is LU[k, k]
                T U_kk = T(1);
                for (int p = LU.row_ptr[k]; p < LU.row_ptr[k+1]; ++p) {
                    if (LU.col_indices[p] == k) {
                        U_kk = LU.values[p];
                        break;
                    }
                }

                LU.values[k_idx] /= U_kk; // L[i,k] = A[i,k] / U[k,k]
                T L_ik = LU.values[k_idx];

                // Update row i: U[i, j] = A[i, j] - L_ik * U[k, j] for j > k
                for (int j_idx = k_idx + 1; j_idx < LU.row_ptr[i+1]; ++j_idx) {
                    int j = LU.col_indices[j_idx];
                    
                    // Find U[k, j] in row k
                    T U_kj = T(0);
                    for (int p = LU.row_ptr[k]; p < LU.row_ptr[k+1]; ++p) {
                        if (LU.col_indices[p] == j) {
                            U_kj = LU.values[p];
                            break;
                        }
                    }
                    if (U_kj != T(0)) {
                        LU.values[j_idx] -= L_ik * U_kj;
                    }
                }
            }
        }
    }

    void apply(const std::vector<T>& r, std::vector<T>& z) const {
        int n = LU.rows();
        z.assign(n, T(0));
        std::vector<T> y(n, T(0));

        // Forward substitution: solve L y = r
        for (int i = 0; i < n; ++i) {
            T sum = r[i];
            for (int idx = LU.row_ptr[i]; idx < LU.row_ptr[i+1]; ++idx) {
                int j = LU.col_indices[idx];
                if (j >= i) break; // strictly lower part
                sum -= LU.values[idx] * y[j];
            }
            y[i] = sum; // L has 1s on diagonal
        }

        // Backward substitution: solve U z = y
        for (int i = n - 1; i >= 0; --i) {
            T sum = y[i];
            T U_ii = T(1);
            for (int idx = LU.row_ptr[i]; idx < LU.row_ptr[i+1]; ++idx) {
                int j = LU.col_indices[idx];
                if (j == i) {
                    U_ii = LU.values[idx];
                } else if (j > i) {
                    sum -= LU.values[idx] * z[j];
                }
            }
            z[i] = sum / U_ii;
        }
    }
};
