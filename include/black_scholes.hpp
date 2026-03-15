#pragma once
#include "common.hpp"

namespace opts {

// ---------------------------------------------------------------------------
// Black-Scholes-Merton pricing & Greeks
// ---------------------------------------------------------------------------
struct BSResult {
    double price = 0.0;
    double delta = 0.0;
    double gamma = 0.0;
    double theta = 0.0;
    double vega  = 0.0;
    double rho   = 0.0;
};

class BlackScholes {
public:
    // Core pricing
    static BSResult calculate(OptionType type, double S, double K,
                              double T, double r, double sigma, double q = 0.0) {
        if (T <= 0.0 || sigma <= 0.0) {
            return expired_value(type, S, K);
        }

        double sqrtT = std::sqrt(T);
        double d1 = (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T)
                     / (sigma * sqrtT);
        double d2 = d1 - sigma * sqrtT;

        double disc  = std::exp(-r * T);
        double qdisc = std::exp(-q * T);

        BSResult res;
        if (type == OptionType::Call) {
            res.price = S * qdisc * norm_cdf(d1) - K * disc * norm_cdf(d2);
            res.delta = qdisc * norm_cdf(d1);
        } else {
            res.price = K * disc * norm_cdf(-d2) - S * qdisc * norm_cdf(-d1);
            res.delta = -qdisc * norm_cdf(-d1);
        }

        double npd1 = norm_pdf(d1);
        res.gamma = qdisc * npd1 / (S * sigma * sqrtT);
        res.vega  = S * qdisc * npd1 * sqrtT * 0.01;  // per 1% vol move
        res.theta = (-(S * sigma * qdisc * npd1) / (2.0 * sqrtT)
                     - r * K * disc * norm_cdf(type == OptionType::Call ? d2 : -d2)
                        * (type == OptionType::Call ? 1.0 : -1.0)
                     + q * S * qdisc * norm_cdf(type == OptionType::Call ? d1 : -d1)
                        * (type == OptionType::Call ? 1.0 : -1.0))
                    / DAYS_PER_YEAR;
        res.rho   = K * T * disc
                     * (type == OptionType::Call ? norm_cdf(d2) : -norm_cdf(-d2))
                     * 0.01;
        return res;
    }

    // Implied volatility via Newton-Raphson
    static double implied_vol(OptionType type, double market_price,
                              double S, double K, double T, double r,
                              double q = 0.0, double init = 0.25,
                              int max_iter = 100, double tol = 1e-8) {
        double sigma = init;
        for (int i = 0; i < max_iter; ++i) {
            auto res = calculate(type, S, K, T, r, sigma, q);
            double diff = res.price - market_price;
            if (std::fabs(diff) < tol) return sigma;
            double vega_raw = res.vega / 0.01; // undo per-1% scaling
            if (std::fabs(vega_raw) < 1e-14) break;
            sigma -= diff / vega_raw;
            if (sigma <= 0.0) sigma = 0.001;
        }
        return sigma;
    }

private:
    static BSResult expired_value(OptionType type, double S, double K) {
        BSResult r;
        if (type == OptionType::Call) {
            r.price = std::max(S - K, 0.0);
            r.delta = (S > K) ? 1.0 : 0.0;
        } else {
            r.price = std::max(K - S, 0.0);
            r.delta = (K > S) ? -1.0 : 0.0;
        }
        return r;
    }
};

} // namespace opts
