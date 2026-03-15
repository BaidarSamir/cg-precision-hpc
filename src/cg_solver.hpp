#pragma once
// ============================================================================
//  Conjugate Gradient Solver — Preconditioned
// ============================================================================

#include "csr_matrix.hpp"
#include <vector>
#include <cmath>
#include <numeric>

template <typename T>
struct CgResult {
    std::vector<T>       x;
    std::vector<double>  residuals;   // ||r_k||_2
    std::vector<double>  true_errors; // ||x_k - x_ref||_2 (if track_drift)
    int                  iterations;
    bool                 converged;
};

template <typename T>
static T dot(const std::vector<T>& a, const std::vector<T>& b) {
    T sum = T(0);
    for (std::size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

template <typename T>
static void axpy(T alpha, const std::vector<T>& x, std::vector<T>& y) {
    for (std::size_t i = 0; i < x.size(); ++i) {
        y[i] += alpha * x[i];
    }
}

template <typename T, typename Preconditioner>
CgResult<T> cg_solve(const CsrMatrix<T>& A,
                     const std::vector<T>& b,
                     const Preconditioner& M,
                     T tol,
                     int max_iter,
                     const std::vector<double>* x_ref = nullptr)
{
    const int n = A.rows();
    CgResult<T> result;
    result.x.assign(n, T(0));
    result.converged = false;
    result.iterations = 0;

    std::vector<T> r(b);
    std::vector<T> z(n);
    M.apply(r, z);
    std::vector<T> p(z);
    std::vector<T> Ap(n);

    T rz = dot(r, z);
    const double b_norm = std::sqrt(static_cast<double>(dot(b, b)));
    const double tol_d = static_cast<double>(tol);

    double r_norm = std::sqrt(static_cast<double>(dot(r, r)));
    result.residuals.push_back(r_norm);

    if (x_ref) {
        double err = 0.0;
        for (int i = 0; i < n; ++i) {
            double diff = static_cast<double>(result.x[i]) - (*x_ref)[i];
            err += diff * diff;
        }
        result.true_errors.push_back(std::sqrt(err));
    }

    for (int k = 0; k < max_iter; ++k) {
        A.spmv(p, Ap);

        T pAp = dot(p, Ap);
        T alpha = rz / pAp;

        axpy(alpha, p, result.x);
        axpy(-alpha, Ap, r);

        r_norm = std::sqrt(static_cast<double>(dot(r, r)));
        result.residuals.push_back(r_norm);

        if (x_ref) {
            double err = 0.0;
            for (int i = 0; i < n; ++i) {
                double diff = static_cast<double>(result.x[i]) - (*x_ref)[i];
                err += diff * diff;
            }
            result.true_errors.push_back(std::sqrt(err));
        }

        result.iterations = k + 1;

        if (b_norm > 0.0 && r_norm / b_norm < tol_d) {
            result.converged = true;
            break;
        }

        M.apply(r, z);
        T rz_new = dot(r, z);

        T beta = rz_new / rz;
        for (int i = 0; i < n; ++i) {
            p[i] = z[i] + beta * p[i];
        }

        rz = rz_new;
    }

    return result;
}
