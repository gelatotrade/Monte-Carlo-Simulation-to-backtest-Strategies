#pragma once
#include "common.hpp"

namespace opts {

// ---------------------------------------------------------------------------
// Geometric Brownian Motion Monte-Carlo simulator
// Generates price paths with stochastic volatility (Heston-lite via regime)
// ---------------------------------------------------------------------------
class MonteCarlo {
public:
    struct PathSet {
        std::vector<std::vector<double>> paths;   // [path_idx][step]
        std::vector<std::vector<double>> vol_paths;
        std::vector<double> time_grid;             // year fractions
        int n_paths = 0;
        int n_steps = 0;
    };

    struct Config {
        double S0          = 100.0;
        double mu          = 0.08;   // drift (expected return)
        double sigma       = 0.20;   // base vol
        double r           = 0.04;
        double q           = 0.0;
        int    n_paths     = 10000;
        int    n_steps     = 252;    // daily
        double T           = 1.0;    // horizon in years

        // Heston-like vol-of-vol parameters (simplified)
        double vol_of_vol  = 0.3;
        double vol_mean    = 0.20;   // long-run vol
        double vol_kappa   = 2.0;    // mean-reversion speed
        double rho         = -0.7;   // spot-vol correlation
    };

    static PathSet simulate(const Config& cfg, unsigned seed = 42) {
        PathSet ps;
        ps.n_paths = cfg.n_paths;
        ps.n_steps = cfg.n_steps;
        ps.paths.resize(cfg.n_paths, std::vector<double>(cfg.n_steps + 1));
        ps.vol_paths.resize(cfg.n_paths, std::vector<double>(cfg.n_steps + 1));
        ps.time_grid.resize(cfg.n_steps + 1);

        double dt = cfg.T / cfg.n_steps;
        double sqrt_dt = std::sqrt(dt);

        for (int t = 0; t <= cfg.n_steps; ++t)
            ps.time_grid[t] = t * dt;

        std::mt19937 gen(seed);
        std::normal_distribution<> ndist(0.0, 1.0);

        for (int p = 0; p < cfg.n_paths; ++p) {
            ps.paths[p][0]     = cfg.S0;
            ps.vol_paths[p][0] = cfg.sigma;

            double S = cfg.S0;
            double v = cfg.sigma * cfg.sigma;  // variance

            for (int t = 1; t <= cfg.n_steps; ++t) {
                double z1 = ndist(gen);
                double z2 = cfg.rho * z1 + std::sqrt(1.0 - cfg.rho * cfg.rho) * ndist(gen);

                // Variance (CIR-like, truncated)
                double v_new = v + cfg.vol_kappa * (cfg.vol_mean * cfg.vol_mean - v) * dt
                               + cfg.vol_of_vol * std::sqrt(std::max(v, 0.0)) * sqrt_dt * z2;
                v = std::max(v_new, 1e-8);

                double sigma_t = std::sqrt(v);
                S *= std::exp((cfg.mu - cfg.q - 0.5 * v) * dt + sigma_t * sqrt_dt * z1);

                ps.paths[p][t]     = S;
                ps.vol_paths[p][t] = sigma_t;
            }
        }
        return ps;
    }

    // Compute percentile paths for fan charts
    static std::vector<std::vector<double>> percentile_paths(
            const PathSet& ps,
            const std::vector<double>& percentiles) {
        int n_pct = static_cast<int>(percentiles.size());
        std::vector<std::vector<double>> result(n_pct,
            std::vector<double>(ps.n_steps + 1));

        for (int t = 0; t <= ps.n_steps; ++t) {
            std::vector<double> cross(ps.n_paths);
            for (int p = 0; p < ps.n_paths; ++p)
                cross[p] = ps.paths[p][t];
            std::sort(cross.begin(), cross.end());

            for (int q = 0; q < n_pct; ++q) {
                double idx = percentiles[q] / 100.0 * (ps.n_paths - 1);
                int lo = static_cast<int>(idx);
                int hi = std::min(lo + 1, ps.n_paths - 1);
                double frac = idx - lo;
                result[q][t] = cross[lo] * (1.0 - frac) + cross[hi] * frac;
            }
        }
        return result;
    }
};

} // namespace opts
