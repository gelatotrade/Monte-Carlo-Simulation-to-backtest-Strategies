#pragma once
#include "common.hpp"
#include "black_scholes.hpp"
#include "strategies.hpp"
#include "vol_regime.hpp"
#include "monte_carlo.hpp"

namespace opts {

// ---------------------------------------------------------------------------
// Trade record
// ---------------------------------------------------------------------------
struct TradeRecord {
    int    day_opened   = 0;
    int    day_closed   = 0;
    double entry_cost   = 0.0;   // net premium at entry
    double exit_value   = 0.0;   // value at exit
    double pnl          = 0.0;
    double spot_at_open = 0.0;
    double spot_at_close = 0.0;
    double iv_at_open   = 0.0;
    double iv_at_close  = 0.0;
    VolRegime regime_at_open = VolRegime::Medium;
    std::string strategy_name;
};

// ---------------------------------------------------------------------------
// Equity curve point
// ---------------------------------------------------------------------------
struct EquityPoint {
    int    day           = 0;
    double equity        = 0.0;
    double benchmark     = 0.0;   // SP500 buy-and-hold
    double drawdown      = 0.0;
    double iv            = 0.0;
    VolRegime regime     = VolRegime::Medium;
};

// ---------------------------------------------------------------------------
// Backtest statistics
// ---------------------------------------------------------------------------
struct BacktestStats {
    double total_return     = 0.0;
    double annualised_return = 0.0;
    double sharpe_ratio     = 0.0;
    double sortino_ratio    = 0.0;
    double max_drawdown     = 0.0;
    double win_rate         = 0.0;
    double profit_factor    = 0.0;
    double avg_pnl          = 0.0;
    double avg_winner       = 0.0;
    double avg_loser        = 0.0;
    int    total_trades     = 0;
    int    winners          = 0;
    int    losers           = 0;
    double benchmark_return = 0.0;
    double alpha            = 0.0;

    // Per-regime stats
    std::map<VolRegime, double> regime_pnl;
    std::map<VolRegime, int>    regime_trades;
};

// ---------------------------------------------------------------------------
// Strategy Signal – tells the backtester when / what to trade
// ---------------------------------------------------------------------------
struct StrategySignal {
    std::string name;
    // Returns legs to open given current market state, empty = no trade
    std::function<std::vector<Leg>(double spot, double iv, double r,
                                   double q, VolRegime regime)> generate;
    int    hold_days  = 30;      // DTE at entry / holding period
    double dte_entry  = 30.0;    // target DTE for option entry
};

// ---------------------------------------------------------------------------
// Backtesting Engine
// ---------------------------------------------------------------------------
class Backtester {
public:
    struct Config {
        double initial_capital   = 100000.0;
        double position_size     = 0.05;   // 5% per trade
        int    rebalance_freq    = 21;     // trade every N days
        double transaction_cost  = 0.65;   // per contract
        int    contracts         = 1;
    };

    // Run backtest on Monte-Carlo simulated paths
    struct BacktestResult {
        BacktestStats                stats;
        std::vector<EquityPoint>     equity_curve;
        std::vector<TradeRecord>     trades;
        VolSurface3D                 surface_3d;    // IV × PnL × Spot
    };

    static BacktestResult run(const MonteCarlo::PathSet& paths,
                              const StrategySignal& strategy,
                              const Config& cfg,
                              const VolRegimeClassifier& regime_clf) {
        BacktestResult result;
        int n_steps = paths.n_steps;

        // Use median path for deterministic backtest, or path 0
        // For a comprehensive run we average across paths
        // Here: iterate the median path for the equity curve,
        //       then aggregate stats across all paths.

        // --- Build IV history from vol paths (use annualised %) -----------
        std::vector<double> spot_path(n_steps + 1);
        std::vector<double> iv_path(n_steps + 1);
        for (int t = 0; t <= n_steps; ++t) {
            // Median across paths at each time step
            std::vector<double> spots(paths.n_paths), vols(paths.n_paths);
            for (int p = 0; p < paths.n_paths; ++p) {
                spots[p] = paths.paths[p][t];
                vols[p]  = paths.vol_paths[p][t] * 100.0; // to VIX-like %
            }
            std::sort(spots.begin(), spots.end());
            std::sort(vols.begin(), vols.end());
            spot_path[t] = spots[paths.n_paths / 2];
            iv_path[t]   = vols[paths.n_paths / 2];
        }

        double equity    = cfg.initial_capital;
        double peak      = equity;
        double benchmark_shares = cfg.initial_capital / spot_path[0];

        std::vector<double> iv_history;
        std::vector<Leg> open_position;
        int  open_day = -1;
        double open_spot = 0.0;
        double open_iv = 0.0;
        VolRegime open_regime = VolRegime::Medium;

        for (int t = 0; t <= n_steps; ++t) {
            double S  = spot_path[t];
            double iv = iv_path[t];
            double prev_iv = (t > 0) ? iv_path[t - 1] : iv;
            double rv = 0.0;
            if (t >= 21) {
                // 21-day realised vol
                double sum_sq = 0.0;
                for (int i = t - 20; i <= t; ++i) {
                    double ret = std::log(spot_path[i] / spot_path[i - 1]);
                    sum_sq += ret * ret;
                }
                rv = std::sqrt(sum_sq / 21.0 * DAYS_PER_YEAR);
            }
            iv_history.push_back(iv);

            auto regime_state = regime_clf.classify(iv, prev_iv, rv, iv_history);

            // --- Check if we need to close an open position ---------------
            if (!open_position.empty()) {
                double elapsed = (t - open_day) / DAYS_PER_YEAR;
                double remaining_T = strategy.dte_entry / DAYS_PER_YEAR - elapsed;

                if (remaining_T <= 1.0 / DAYS_PER_YEAR || t == n_steps) {
                    // Close position
                    auto exit_pnl = StrategyFactory::evaluate(
                        open_position, S, 0.04, iv / 100.0, std::max(remaining_T, 0.0));

                    double entry_cost = 0.0;
                    for (auto& leg : open_position)
                        entry_cost += leg.quantity * leg.entry_price;

                    double trade_pnl = (exit_pnl.gross_pnl - entry_cost)
                                       * cfg.contracts * 100.0;  // per contract = 100 shares
                    trade_pnl -= cfg.transaction_cost * cfg.contracts * open_position.size() * 2;

                    equity += trade_pnl;

                    TradeRecord rec;
                    rec.day_opened    = open_day;
                    rec.day_closed    = t;
                    rec.entry_cost    = entry_cost;
                    rec.exit_value    = exit_pnl.gross_pnl;
                    rec.pnl           = trade_pnl;
                    rec.spot_at_open  = open_spot;
                    rec.spot_at_close = S;
                    rec.iv_at_open    = open_iv;
                    rec.iv_at_close   = iv;
                    rec.regime_at_open = open_regime;
                    rec.strategy_name  = strategy.name;
                    result.trades.push_back(rec);

                    open_position.clear();
                }
            }

            // --- Open new position if conditions met ----------------------
            if (open_position.empty() && t % cfg.rebalance_freq == 0 && t < n_steps - strategy.hold_days) {
                double r = 0.04;
                double q = 0.0;
                double T_entry = strategy.dte_entry / DAYS_PER_YEAR;

                auto legs = strategy.generate(S, iv / 100.0, r, q, regime_state.regime);
                if (!legs.empty()) {
                    open_position = legs;
                    open_day = t;
                    open_spot = S;
                    open_iv = iv;
                    open_regime = regime_state.regime;
                }
            }

            // --- Record equity curve --------------------------------------
            peak = std::max(peak, equity);
            double dd = (peak > 0) ? (peak - equity) / peak : 0.0;

            EquityPoint ep;
            ep.day       = t;
            ep.equity    = equity;
            ep.benchmark = benchmark_shares * S;
            ep.drawdown  = dd;
            ep.iv        = iv;
            ep.regime    = regime_state.regime;
            result.equity_curve.push_back(ep);

            // --- 3D surface point: IV × PnL × SP500 ----------------------
            if (t % 5 == 0) {  // sample every 5 days to reduce data volume
                double cum_pnl = equity - cfg.initial_capital;
                result.surface_3d.add(iv, cum_pnl, S, regime_state.regime);
            }
        }

        // --- Compute statistics -------------------------------------------
        result.stats = compute_stats(result.trades, result.equity_curve,
                                     cfg.initial_capital, n_steps);
        return result;
    }

private:
    static BacktestStats compute_stats(const std::vector<TradeRecord>& trades,
                                       const std::vector<EquityPoint>& curve,
                                       double initial_capital, int n_steps) {
        BacktestStats s;
        s.total_trades = static_cast<int>(trades.size());

        if (trades.empty()) return s;

        double gross_profit = 0, gross_loss = 0;
        for (auto& t : trades) {
            s.avg_pnl += t.pnl;
            if (t.pnl > 0) {
                s.winners++;
                gross_profit += t.pnl;
                s.avg_winner += t.pnl;
            } else {
                s.losers++;
                gross_loss += std::fabs(t.pnl);
                s.avg_loser += t.pnl;
            }
            s.regime_pnl[t.regime_at_open] += t.pnl;
            s.regime_trades[t.regime_at_open]++;
        }

        s.total_return = (curve.back().equity - initial_capital) / initial_capital;
        double years   = n_steps / DAYS_PER_YEAR;
        s.annualised_return = std::pow(1.0 + s.total_return, 1.0 / years) - 1.0;

        s.benchmark_return = (curve.back().benchmark - initial_capital) / initial_capital;
        s.alpha = s.annualised_return
                  - (std::pow(1.0 + s.benchmark_return, 1.0 / years) - 1.0);

        s.win_rate = (s.total_trades > 0)
                     ? 100.0 * s.winners / s.total_trades : 0.0;
        s.profit_factor = (gross_loss > 0) ? gross_profit / gross_loss : 999.0;
        s.avg_pnl    /= s.total_trades;
        if (s.winners > 0) s.avg_winner /= s.winners;
        if (s.losers  > 0) s.avg_loser  /= s.losers;

        // Max drawdown
        s.max_drawdown = 0;
        for (auto& ep : curve)
            s.max_drawdown = std::max(s.max_drawdown, ep.drawdown);

        // Sharpe & Sortino (daily returns → annualised)
        std::vector<double> daily_ret;
        for (size_t i = 1; i < curve.size(); ++i) {
            if (curve[i - 1].equity > 0)
                daily_ret.push_back(curve[i].equity / curve[i - 1].equity - 1.0);
        }
        if (!daily_ret.empty()) {
            double mean = std::accumulate(daily_ret.begin(), daily_ret.end(), 0.0)
                          / daily_ret.size();
            double var = 0, downvar = 0;
            for (double r : daily_ret) {
                var += (r - mean) * (r - mean);
                if (r < 0) downvar += r * r;
            }
            double std_dev = std::sqrt(var / daily_ret.size());
            double down_dev = std::sqrt(downvar / daily_ret.size());
            s.sharpe_ratio  = (std_dev > 0) ? mean / std_dev * std::sqrt(DAYS_PER_YEAR) : 0.0;
            s.sortino_ratio = (down_dev > 0) ? mean / down_dev * std::sqrt(DAYS_PER_YEAR) : 0.0;
        }

        return s;
    }
};

} // namespace opts
