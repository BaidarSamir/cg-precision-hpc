// ============================================================================
//  CG Precision HPC — Main Driver
// ============================================================================

#include "csr_matrix.hpp"
#include "laplacian.hpp"
#include "cg_solver.hpp"
#include "preconditioners.hpp"

#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>

namespace fs = std::filesystem;

struct Config {
    int         grid_size = 10;
    double      tol       = 1e-8;
    int         max_iter  = 1000;
    std::string precision = "double";
    std::string precon    = "none";
    bool        track_drift = false;
    bool        tradeoff_sweep = false;
    bool        show_help = false;
    bool        valid = true;
    std::string error_message;
};

void print_usage(const char* exe_name) {
    std::cout
        << "Usage: " << exe_name << " [options]\n\n"
        << "Options:\n"
        << "  --grid-size <int>      Grid edge length n for n x n problem (default: 10)\n"
        << "  --tol <float>          Relative residual tolerance (default: 1e-8)\n"
        << "  --max-iter <int>       Maximum CG iterations (default: 1000)\n"
        << "  --precision <string>   Solver precision: float | double (default: double)\n"
        << "  --precon <string>      Preconditioner: none | jacobi | ilu0 (default: none)\n"
        << "  --track-drift          Run float/double true-error drift analysis\n"
        << "  --tradeoff-sweep       Run precision/runtime tolerance sweep\n"
        << "  --help                 Show this message\n\n"
        << "Notes:\n"
        << "  - --track-drift and --tradeoff-sweep ignore --precision and --precon.\n"
        << "  - Output CSV files are written to ./data.\n";
}

Config parse_args(int argc, char* argv[]) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help") {
            cfg.show_help = true;
            return cfg;
        } else if (arg == "--grid-size") {
            if (i + 1 >= argc) {
                cfg.valid = false;
                cfg.error_message = "Missing value for --grid-size";
                return cfg;
            }
            cfg.grid_size = std::atoi(argv[++i]);
        } else if (arg == "--tol") {
            if (i + 1 >= argc) {
                cfg.valid = false;
                cfg.error_message = "Missing value for --tol";
                return cfg;
            }
            cfg.tol = std::atof(argv[++i]);
        } else if (arg == "--max-iter") {
            if (i + 1 >= argc) {
                cfg.valid = false;
                cfg.error_message = "Missing value for --max-iter";
                return cfg;
            }
            cfg.max_iter = std::atoi(argv[++i]);
        } else if (arg == "--precision") {
            if (i + 1 >= argc) {
                cfg.valid = false;
                cfg.error_message = "Missing value for --precision";
                return cfg;
            }
            cfg.precision = argv[++i];
        } else if (arg == "--precon") {
            if (i + 1 >= argc) {
                cfg.valid = false;
                cfg.error_message = "Missing value for --precon";
                return cfg;
            }
            cfg.precon = argv[++i];
        }
        else if (arg == "--track-drift") cfg.track_drift = true;
        else if (arg == "--tradeoff-sweep") cfg.tradeoff_sweep = true;
        else {
            cfg.valid = false;
            cfg.error_message = "Unknown argument: " + arg;
            return cfg;
        }
    }

    if (cfg.grid_size <= 0) {
        cfg.valid = false;
        cfg.error_message = "--grid-size must be > 0";
        return cfg;
    }
    if (cfg.tol <= 0.0) {
        cfg.valid = false;
        cfg.error_message = "--tol must be > 0";
        return cfg;
    }
    if (cfg.max_iter <= 0) {
        cfg.valid = false;
        cfg.error_message = "--max-iter must be > 0";
        return cfg;
    }
    if (cfg.precision != "float" && cfg.precision != "double") {
        cfg.valid = false;
        cfg.error_message = "--precision must be one of: float, double";
        return cfg;
    }
    if (cfg.precon != "none" && cfg.precon != "jacobi" && cfg.precon != "ilu0") {
        cfg.valid = false;
        cfg.error_message = "--precon must be one of: none, jacobi, ilu0";
        return cfg;
    }

    return cfg;
}

void run_drift(const Config& cfg) {
    const int n = cfg.grid_size;
    const int N = n * n;
    
    std::cout << "[Drift Analysis] Grid: " << n << "x" << n << "\n";
    
    auto Ad = build_laplacian_2d<double>(n);
    std::vector<double> bd(N, 1.0);
    Ilu0Preconditioner<double> Md(Ad);
    
    std::cout << "1. Computing reference solution (double, tol 1e-12)...\n";
    auto ref_result = cg_solve(Ad, bd, Md, 1e-12, 5000);
    std::vector<double> x_ref = ref_result.x;
    
    std::cout << "2. Running float tracking drift...\n";
    auto Af = build_laplacian_2d<float>(n);
    std::vector<float> bf(N, 1.0f);
    Ilu0Preconditioner<float> Mf(Af);
    auto res_f = cg_solve(Af, bf, Mf, 1e-8f, cfg.max_iter, &x_ref);
    
    std::cout << "3. Running double tracking drift...\n";
    auto res_d = cg_solve(Ad, bd, Md, 1e-8, cfg.max_iter, &x_ref);
    
    fs::create_directories("data");
    std::ofstream fout("data/drift_comparison_" + std::to_string(n) + ".csv");
    fout << "iteration,true_error_float,true_error_double\n";
    
    std::size_t max_its = std::max(res_f.true_errors.size(), res_d.true_errors.size());
    for (std::size_t i = 0; i < max_its; ++i) {
        fout << i << ",";
        if (i < res_f.true_errors.size()) fout << std::scientific << res_f.true_errors[i];
        fout << ",";
        if (i < res_d.true_errors.size()) fout << std::scientific << res_d.true_errors[i];
        fout << "\n";
    }
    std::cout << "Drift data written to data/drift_comparison_" << n << ".csv\n";
}

void run_tradeoff(const Config& cfg) {
    const int n = cfg.grid_size;
    const int N = n * n;
    
    std::vector<double> tols = {1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10};
    
    fs::create_directories("data");
    std::ofstream fout("data/tradeoff_" + std::to_string(n) + ".csv");
    fout << "precision,target_tol,actual_iterations,wall_time_ms,final_residual\n";
    
    auto run_sweep_f = [&]() {
        auto A = build_laplacian_2d<float>(n);
        std::vector<float> b(N, 1.0f);
        Ilu0Preconditioner<float> M(A);
        for (double t : tols) {
            auto t_start = std::chrono::high_resolution_clock::now();
            auto res = cg_solve(A, b, M, static_cast<float>(t), cfg.max_iter);
            auto t_end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
            
            fout << "float," << std::scientific << t << "," 
                 << res.iterations << "," << std::fixed << ms << "," 
                 << std::scientific << res.residuals.back() << "\n";
            std::cout << "float tol " << t << " -> " << res.iterations << " iters\n";
        }
    };
    
    auto run_sweep_d = [&]() {
        auto A = build_laplacian_2d<double>(n);
        std::vector<double> b(N, 1.0);
        Ilu0Preconditioner<double> M(A);
        for (double t : tols) {
            auto t_start = std::chrono::high_resolution_clock::now();
            auto res = cg_solve(A, b, M, t, cfg.max_iter);
            auto t_end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
            
            fout << "double," << std::scientific << t << "," 
                 << res.iterations << "," << std::fixed << ms << "," 
                 << std::scientific << res.residuals.back() << "\n";
            std::cout << "double tol " << t << " -> " << res.iterations << " iters\n";
        }
    };
    
    run_sweep_f();
    run_sweep_d();
    std::cout << "Tradeoff data written to data/tradeoff_" << n << ".csv\n";
}

template <typename T>
void run_cg(const Config& cfg) {
    const int n = cfg.grid_size;
    const int N = n * n;

    auto A = build_laplacian_2d<T>(n);
    std::vector<T> b(N, T(1));

    CgResult<T> result;
    
    if (cfg.precon == "none") {
        NoPreconditioner<T> M(A);
        result = cg_solve(A, b, M, static_cast<T>(cfg.tol), cfg.max_iter);
    } else if (cfg.precon == "jacobi") {
        JacobiPreconditioner<T> M(A);
        result = cg_solve(A, b, M, static_cast<T>(cfg.tol), cfg.max_iter);
    } else if (cfg.precon == "ilu0") {
        Ilu0Preconditioner<T> M(A);
        result = cg_solve(A, b, M, static_cast<T>(cfg.tol), cfg.max_iter);
    }

    fs::create_directories("data");
    std::string filename = "data/convergence_" + cfg.precision 
                         + "_" + cfg.precon + "_" + std::to_string(n) + ".csv";

    std::ofstream fout(filename);
    if (fout.is_open()) {
        fout << "iteration,residual_norm\n";
        for (std::size_t i = 0; i < result.residuals.size(); ++i) {
            fout << i << "," << std::scientific << std::setprecision(15)
                 << result.residuals[i] << "\n";
        }
    }
}

int main(int argc, char* argv[]) {
    Config cfg = parse_args(argc, argv);

    if (cfg.show_help) {
        print_usage(argv[0]);
        return 0;
    }
    if (!cfg.valid) {
        std::cerr << "Error: " << cfg.error_message << "\n\n";
        print_usage(argv[0]);
        return 1;
    }

    if (cfg.track_drift) {
        run_drift(cfg);
    } else if (cfg.tradeoff_sweep) {
        run_tradeoff(cfg);
    } else {
        if (cfg.precision == "float") run_cg<float>(cfg);
        else if (cfg.precision == "double") run_cg<double>(cfg);
    }

    return 0;
}
