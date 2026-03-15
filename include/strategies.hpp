#pragma once
#include "common.hpp"
#include "black_scholes.hpp"
#include <utility>

namespace opts {

// ---------------------------------------------------------------------------
// Option Leg – building block for multi-leg strategies
// ---------------------------------------------------------------------------
struct Leg {
    OptionType type;
    double     strike;
    double     expiry_T;     // time-to-expiry when opened (years)
    int        quantity;      // +1 long, -1 short
    double     entry_price;   // premium paid / received per unit
};

// ---------------------------------------------------------------------------
// Strategy result at a single evaluation point
// ---------------------------------------------------------------------------
struct StrategyPnL {
    double gross_pnl       = 0.0;
    double net_pnl         = 0.0;   // after entry cost
    double net_premium      = 0.0;   // total premium collected/paid at entry
    double delta            = 0.0;
    double gamma            = 0.0;
    double vega             = 0.0;
    double theta            = 0.0;
};

// ---------------------------------------------------------------------------
// Strategy Factory – constructs common options strategies
// ---------------------------------------------------------------------------
class StrategyFactory {
public:
    // ----- Straddle (Vega Expansion play) ----------------------------------
    static std::vector<Leg> straddle(double atm_strike, double T,
                                     double S, double r, double sigma,
                                     double q = 0.0, bool is_long = true) {
        int qty = is_long ? 1 : -1;
        auto call = BlackScholes::calculate(OptionType::Call, S, atm_strike, T, r, sigma, q);
        auto put  = BlackScholes::calculate(OptionType::Put,  S, atm_strike, T, r, sigma, q);
        return {
            {OptionType::Call, atm_strike, T, qty, call.price},
            {OptionType::Put,  atm_strike, T, qty, put.price}
        };
    }

    // ----- Strangle ---------------------------------------------------------
    static std::vector<Leg> strangle(double call_strike, double put_strike,
                                     double T, double S, double r, double sigma,
                                     double q = 0.0, bool is_long = true) {
        int qty = is_long ? 1 : -1;
        auto call = BlackScholes::calculate(OptionType::Call, S, call_strike, T, r, sigma, q);
        auto put  = BlackScholes::calculate(OptionType::Put,  S, put_strike,  T, r, sigma, q);
        return {
            {OptionType::Call, call_strike, T, qty, call.price},
            {OptionType::Put,  put_strike,  T, qty, put.price}
        };
    }

    // ----- Butterfly Spread (long call butterfly) --------------------------
    static std::vector<Leg> butterfly(double lower, double middle, double upper,
                                      double T, double S, double r, double sigma,
                                      double q = 0.0) {
        auto c_lo  = BlackScholes::calculate(OptionType::Call, S, lower,  T, r, sigma, q);
        auto c_mid = BlackScholes::calculate(OptionType::Call, S, middle, T, r, sigma, q);
        auto c_up  = BlackScholes::calculate(OptionType::Call, S, upper,  T, r, sigma, q);
        return {
            {OptionType::Call, lower,  T,  1, c_lo.price},
            {OptionType::Call, middle, T, -2, c_mid.price},
            {OptionType::Call, upper,  T,  1, c_up.price}
        };
    }

    // ----- Iron Condor (short) ---------------------------------------------
    static std::vector<Leg> iron_condor(double put_long_K, double put_short_K,
                                        double call_short_K, double call_long_K,
                                        double T, double S, double r, double sigma,
                                        double q = 0.0) {
        auto pl = BlackScholes::calculate(OptionType::Put,  S, put_long_K,   T, r, sigma, q);
        auto ps = BlackScholes::calculate(OptionType::Put,  S, put_short_K,  T, r, sigma, q);
        auto cs = BlackScholes::calculate(OptionType::Call, S, call_short_K, T, r, sigma, q);
        auto cl = BlackScholes::calculate(OptionType::Call, S, call_long_K,  T, r, sigma, q);
        return {
            {OptionType::Put,  put_long_K,   T,  1, pl.price},   // long OTM put
            {OptionType::Put,  put_short_K,  T, -1, ps.price},   // short put
            {OptionType::Call, call_short_K, T, -1, cs.price},   // short call
            {OptionType::Call, call_long_K,  T,  1, cl.price}    // long OTM call
        };
    }

    // ----- Iron Butterfly --------------------------------------------------
    static std::vector<Leg> iron_butterfly(double lower, double middle, double upper,
                                           double T, double S, double r, double sigma,
                                           double q = 0.0) {
        auto pl = BlackScholes::calculate(OptionType::Put,  S, lower,  T, r, sigma, q);
        auto ps = BlackScholes::calculate(OptionType::Put,  S, middle, T, r, sigma, q);
        auto cs = BlackScholes::calculate(OptionType::Call, S, middle, T, r, sigma, q);
        auto cl = BlackScholes::calculate(OptionType::Call, S, upper,  T, r, sigma, q);
        return {
            {OptionType::Put,  lower,  T,  1, pl.price},
            {OptionType::Put,  middle, T, -1, ps.price},
            {OptionType::Call, middle, T, -1, cs.price},
            {OptionType::Call, upper,  T,  1, cl.price}
        };
    }

    // ----- Calendar Spread -------------------------------------------------
    static std::vector<Leg> calendar_spread(double strike, double T_near, double T_far,
                                            double S, double r, double sigma,
                                            double q = 0.0) {
        auto c_near = BlackScholes::calculate(OptionType::Call, S, strike, T_near, r, sigma, q);
        auto c_far  = BlackScholes::calculate(OptionType::Call, S, strike, T_far,  r, sigma, q);
        return {
            {OptionType::Call, strike, T_near, -1, c_near.price},  // short near
            {OptionType::Call, strike, T_far,   1, c_far.price}    // long far
        };
    }

    // ----- Ratio Spread (1x2 call) ----------------------------------------
    static std::vector<Leg> ratio_spread(double lower, double upper,
                                         double T, double S, double r, double sigma,
                                         double q = 0.0) {
        auto c_lo = BlackScholes::calculate(OptionType::Call, S, lower, T, r, sigma, q);
        auto c_up = BlackScholes::calculate(OptionType::Call, S, upper, T, r, sigma, q);
        return {
            {OptionType::Call, lower, T,  1, c_lo.price},
            {OptionType::Call, upper, T, -2, c_up.price}
        };
    }

    // -----------------------------------------------------------------------
    // Evaluate a multi-leg position at a given spot / vol / time
    // -----------------------------------------------------------------------
    static StrategyPnL evaluate(const std::vector<Leg>& legs,
                                double S, double r, double sigma,
                                double remaining_T, double q = 0.0) {
        StrategyPnL result;
        for (auto& leg : legs) {
            double T_rem = std::max(remaining_T, 0.0);
            auto bs = BlackScholes::calculate(leg.type, S, leg.strike, T_rem, r, sigma, q);

            result.gross_pnl += leg.quantity * bs.price;
            result.net_premium += leg.quantity * leg.entry_price;
            result.delta += leg.quantity * bs.delta;
            result.gamma += leg.quantity * bs.gamma;
            result.vega  += leg.quantity * bs.vega;
            result.theta += leg.quantity * bs.theta;
        }
        result.net_pnl = result.gross_pnl - result.net_premium;
        return result;
    }
};

} // namespace opts
