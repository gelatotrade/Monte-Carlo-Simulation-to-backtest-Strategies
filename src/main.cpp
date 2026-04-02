#include "common.hpp"
#include "black_scholes.hpp"
#include "monte_carlo.hpp"
#include "strategies.hpp"
#include "vol_regime.hpp"
#include "backtester.hpp"

using namespace opts;

// ---------------------------------------------------------------------------
// Helper: print backtest report
// ---------------------------------------------------------------------------
static void print_report(const std::string& name,
                         const Backtester::BacktestResult& res) {
    auto& s = res.stats;
    std::cout << "\n========================================================\n";
    std::cout << "  BACKTEST REPORT: " << name << "\n";
    std::cout << "========================================================\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Total Return:       " << s.total_return * 100.0 << " %\n";
    std::cout << "  Annualised Return:  " << s.annualised_return * 100.0 << " %\n";
    std::cout << "  Benchmark Return:   " << s.benchmark_return * 100.0 << " % (S&P 500)\n";
    std::cout << "  Alpha:              " << s.alpha * 100.0 << " %\n";
    std::cout << "  Sharpe Ratio:       " << s.sharpe_ratio << "\n";
    std::cout << "  Sortino Ratio:      " << s.sortino_ratio << "\n";
    std::cout << "  Max Drawdown:       " << s.max_drawdown * 100.0 << " %\n";
    std::cout << "  Win Rate:           " << s.win_rate << " %\n";
    std::cout << "  Profit Factor:      " << s.profit_factor << "\n";
    std::cout << "  Total Trades:       " << s.total_trades << "\n";
    std::cout << "  Avg PnL / Trade:    $" << s.avg_pnl << "\n";
    std::cout << "  Avg Winner:         $" << s.avg_winner << "\n";
    std::cout << "  Avg Loser:          $" << s.avg_loser << "\n";
    std::cout << "\n  --- Performance by Volatility Regime ---\n";
    for (auto& [regime, pnl] : s.regime_pnl) {
        int trades = s.regime_trades.at(regime);
        std::cout << "    " << std::setw(12) << VolRegimeClassifier::regime_name(regime)
                  << ":  PnL = $" << std::setw(10) << pnl
                  << "   Trades = " << trades << "\n";
    }
    std::cout << "========================================================\n\n";
}

// ---------------------------------------------------------------------------
// Helper: export equity curve CSV
// ---------------------------------------------------------------------------
static void export_equity_csv(const std::string& filename,
                              const std::vector<EquityPoint>& curve) {
    std::ofstream out(filename);
    out << "Day,Equity,Benchmark,Drawdown,IV,Regime\n";
    for (auto& ep : curve) {
        out << ep.day << "," << ep.equity << "," << ep.benchmark << ","
            << ep.drawdown << "," << ep.iv << ","
            << VolRegimeClassifier::regime_name(ep.regime) << "\n";
    }
}

// ---------------------------------------------------------------------------
// Helper: export trade log CSV
// ---------------------------------------------------------------------------
static void export_trades_csv(const std::string& filename,
                              const std::vector<TradeRecord>& trades) {
    std::ofstream out(filename);
    out << "Strategy,DayOpened,DayClosed,SpotOpen,SpotClose,IVOpen,IVClose,"
        << "EntryCost,ExitValue,PnL,Regime\n";
    for (auto& t : trades) {
        out << t.strategy_name << "," << t.day_opened << "," << t.day_closed << ","
            << t.spot_at_open << "," << t.spot_at_close << ","
            << t.iv_at_open << "," << t.iv_at_close << ","
            << t.entry_cost << "," << t.exit_value << "," << t.pnl << ","
            << VolRegimeClassifier::regime_name(t.regime_at_open) << "\n";
    }
}

// ===== STRATEGY DEFINITIONS ==============================================

StrategySignal make_vega_expansion_straddle() {
    StrategySignal sig;
    sig.name      = "Vega Expansion Straddle";
    sig.hold_days = 30;
    sig.dte_entry = 30.0;
    sig.generate  = [](double S, double iv, double r, double q, VolRegime regime)
                    -> std::vector<Leg> {
        // Long straddle when vol is low, expanding, or medium → bet on vega increase
        if (regime == VolRegime::Low || regime == VolRegime::Expansion || regime == VolRegime::Medium) {
            double atm = std::round(S);
            return StrategyFactory::straddle(atm, 30.0 / DAYS_PER_YEAR,
                                             S, r, iv, q, true);
        }
        return {};
    };
    return sig;
}

StrategySignal make_iron_condor() {
    StrategySignal sig;
    sig.name      = "Short Iron Condor";
    sig.hold_days = 30;
    sig.dte_entry = 30.0;
    sig.generate  = [](double S, double iv, double r, double q, VolRegime regime)
                    -> std::vector<Leg> {
        // Sell iron condors in medium/high vol (collect premium)
        if (regime == VolRegime::Medium || regime == VolRegime::High) {
            double width = S * 0.05;  // 5% wing width
            double put_short  = std::round(S - width);
            double put_long   = std::round(S - 2.0 * width);
            double call_short = std::round(S + width);
            double call_long  = std::round(S + 2.0 * width);
            return StrategyFactory::iron_condor(put_long, put_short,
                                                call_short, call_long,
                                                30.0 / DAYS_PER_YEAR,
                                                S, r, iv, q);
        }
        return {};
    };
    return sig;
}

StrategySignal make_butterfly() {
    StrategySignal sig;
    sig.name      = "Long Butterfly";
    sig.hold_days = 21;
    sig.dte_entry = 21.0;
    sig.generate  = [](double S, double iv, double r, double q, VolRegime regime)
                    -> std::vector<Leg> {
        // Butterfly in medium vol, expecting pinning near current level
        if (regime == VolRegime::Medium || regime == VolRegime::Crush) {
            double width = S * 0.03;
            double lower  = std::round(S - width);
            double middle = std::round(S);
            double upper  = std::round(S + width);
            return StrategyFactory::butterfly(lower, middle, upper,
                                             21.0 / DAYS_PER_YEAR,
                                             S, r, iv, q);
        }
        return {};
    };
    return sig;
}

StrategySignal make_iron_butterfly() {
    StrategySignal sig;
    sig.name      = "Short Iron Butterfly";
    sig.hold_days = 30;
    sig.dte_entry = 30.0;
    sig.generate  = [](double S, double iv, double r, double q, VolRegime regime)
                    -> std::vector<Leg> {
        // Iron butterfly in high/medium vol → collect maximum premium
        if (regime == VolRegime::High || regime == VolRegime::Crush || regime == VolRegime::Medium) {
            double width = S * 0.05;
            double lower  = std::round(S - width);
            double middle = std::round(S);
            double upper  = std::round(S + width);
            return StrategyFactory::iron_butterfly(lower, middle, upper,
                                                   30.0 / DAYS_PER_YEAR,
                                                   S, r, iv, q);
        }
        return {};
    };
    return sig;
}

StrategySignal make_calendar_spread() {
    StrategySignal sig;
    sig.name      = "Calendar Spread";
    sig.hold_days = 21;
    sig.dte_entry = 21.0;
    sig.generate  = [](double S, double iv, double r, double q, VolRegime regime)
                    -> std::vector<Leg> {
        // Calendar spread – profit from term-structure normalisation
        if (regime == VolRegime::Medium || regime == VolRegime::Low) {
            double atm = std::round(S);
            return StrategyFactory::calendar_spread(atm,
                                                    21.0 / DAYS_PER_YEAR,
                                                    60.0 / DAYS_PER_YEAR,
                                                    S, r, iv, q);
        }
        return {};
    };
    return sig;
}

StrategySignal make_short_strangle() {
    StrategySignal sig;
    sig.name      = "Short Strangle";
    sig.hold_days = 30;
    sig.dte_entry = 30.0;
    sig.generate  = [](double S, double iv, double r, double q, VolRegime regime)
                    -> std::vector<Leg> {
        // Short strangle in high/medium IV environments
        if (regime == VolRegime::High || regime == VolRegime::Medium) {
            double call_K = std::round(S * 1.05);
            double put_K  = std::round(S * 0.95);
            return StrategyFactory::strangle(call_K, put_K,
                                             30.0 / DAYS_PER_YEAR,
                                             S, r, iv, q, false);
        }
        return {};
    };
    return sig;
}

// ---------------------------------------------------------------------------
// Helper: print ARIMA-enhanced backtest report
// ---------------------------------------------------------------------------
static void print_arima_report(const std::string& name,
                               const Backtester::ARIMABacktestResult& res) {
    auto& s = res.stats;
    std::cout << "\n========================================================\n";
    std::cout << "  ARIMA-ENHANCED REPORT: " << name << "\n";
    std::cout << "========================================================\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Total Return:       " << s.total_return * 100.0 << " %\n";
    std::cout << "  Annualised Return:  " << s.annualised_return * 100.0 << " %\n";
    std::cout << "  Alpha:              " << s.alpha * 100.0 << " %\n";
    std::cout << "  Sharpe Ratio:       " << s.sharpe_ratio << "\n";
    std::cout << "  Sortino Ratio:      " << s.sortino_ratio << "\n";
    std::cout << "  Max Drawdown:       " << s.max_drawdown * 100.0 << " %\n";
    std::cout << "  Win Rate:           " << s.win_rate << " %\n";
    std::cout << "  Total Trades:       " << s.total_trades << "\n";
    std::cout << "\n  --- ARIMA Signal Stats ---\n";
    std::cout << "    Entries taken:    " << res.arima_entries_taken << "\n";
    std::cout << "    Entries skipped:  " << res.arima_entries_skipped
              << " (IV direction filter)\n";
    std::cout << "    Early exits:      " << res.arima_early_exits
              << " (adverse regime forecast)\n";

    if (res.iv_model.n_obs > 0) {
        std::cout << "\n  --- IV ARIMA Model ---\n";
        std::cout << "    Order:   ARIMA(" << res.iv_model.p << ","
                  << res.iv_model.d << "," << res.iv_model.q << ")\n";
        std::cout << "    AIC:     " << std::setprecision(1) << res.iv_model.aic << "\n";
        std::cout << "    BIC:     " << res.iv_model.bic << "\n";
        std::cout << "    RMSE:    " << std::setprecision(4) << res.iv_model.rmse << "\n";
        std::cout << "    AR:      [";
        for (size_t i = 0; i < res.iv_model.ar_coeffs.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << std::setprecision(4) << res.iv_model.ar_coeffs[i];
        }
        std::cout << "]\n";
        std::cout << "    MA:      [";
        for (size_t i = 0; i < res.iv_model.ma_coeffs.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << std::setprecision(4) << res.iv_model.ma_coeffs[i];
        }
        std::cout << "]\n";
    }
    std::cout << "========================================================\n\n";
}

// ---------------------------------------------------------------------------
// Helper: export ARIMA forecast CSV
// ---------------------------------------------------------------------------
static void export_arima_csv(const std::string& filename,
                             const Backtester::ARIMABacktestResult& res) {
    std::ofstream out(filename);
    out << "Day,IV_Forecast,IV_Lower95,IV_Upper95,IV_Direction,IV_Confidence,"
        << "Spot_Forecast,Spot_Lower95,Spot_Upper95,Spot_Direction\n";
    for (size_t i = 0; i < res.iv_forecasts.size(); ++i) {
        auto& ivf = res.iv_forecasts[i];
        auto& spf = (i < res.spot_forecasts.size()) ? res.spot_forecasts[i] : ivf;
        int day = (i < res.forecast_days.size()) ? res.forecast_days[i] : 0;
        out << day << ","
            << std::fixed << std::setprecision(4)
            << ivf.point_forecast << "," << ivf.lower_95 << "," << ivf.upper_95 << ","
            << ivf.direction << "," << ivf.confidence << ","
            << std::setprecision(2)
            << spf.point_forecast << "," << spf.lower_95 << "," << spf.upper_95 << ","
            << spf.direction << "\n";
    }
}

// =========================================================================
// MAIN
// =========================================================================
int main() {
    std::cout << "============================================================\n";
    std::cout << "   Options Strategy Backtesting Engine (Monte Carlo + ARIMA)\n";
    std::cout << "   S&P 500 Benchmark  |  Volatility Regime Analysis        \n";
    std::cout << "============================================================\n\n";

    // --- 1. Monte-Carlo simulation ----------------------------------------
    std::cout << "[1/5] Running Monte-Carlo simulation (10,000 paths × 504 steps)...\n";
    MonteCarlo::Config mc_cfg;
    mc_cfg.S0        = 4500.0;      // approximate SPX level
    mc_cfg.mu        = 0.08;
    mc_cfg.sigma     = 0.18;
    mc_cfg.r         = 0.04;
    mc_cfg.q         = 0.015;
    mc_cfg.n_paths   = 10000;
    mc_cfg.n_steps   = 504;         // 2 years of trading days
    mc_cfg.T         = 2.0;
    mc_cfg.vol_of_vol = 0.35;
    mc_cfg.vol_mean   = 0.18;
    mc_cfg.vol_kappa  = 3.0;
    mc_cfg.rho        = -0.70;

    auto paths = MonteCarlo::simulate(mc_cfg, 42);
    std::cout << "   Done. Paths generated.\n\n";

    // --- 2. Volatility regime classifier ----------------------------------
    VolRegimeClassifier regime_clf;
    regime_clf.low_iv_threshold  = 16.5;
    regime_clf.high_iv_threshold = 18.0;
    regime_clf.expansion_rate    =  1.0;
    regime_clf.crush_rate        = -1.0;

    // --- 3. Define strategies --------------------------------------------
    std::cout << "[2/5] Configuring strategies...\n";
    std::vector<StrategySignal> strategies = {
        make_vega_expansion_straddle(),
        make_iron_condor(),
        make_butterfly(),
        make_iron_butterfly(),
        make_calendar_spread(),
        make_short_strangle()
    };
    std::cout << "   " << strategies.size() << " strategies configured.\n\n";

    // --- 4. Run backtests ------------------------------------------------
    std::cout << "[3/5] Running backtests...\n";
    Backtester::Config bt_cfg;
    bt_cfg.initial_capital = 100000.0;
    bt_cfg.position_size   = 0.05;
    bt_cfg.rebalance_freq  = 21;
    bt_cfg.transaction_cost = 0.65;
    bt_cfg.contracts        = 10;

    std::vector<Backtester::BacktestResult> results;
    for (auto& strat : strategies) {
        auto res = Backtester::run(paths, strat, bt_cfg, regime_clf);
        print_report(strat.name, res);
        results.push_back(std::move(res));
    }

    // --- 5. Export data ---------------------------------------------------
    std::cout << "[4/5] Exporting results to output/...\n";
    for (size_t i = 0; i < strategies.size(); ++i) {
        std::string prefix = "output/" + std::to_string(i) + "_";
        // Sanitise name
        std::string sname = strategies[i].name;
        std::replace(sname.begin(), sname.end(), ' ', '_');
        prefix += sname;

        export_equity_csv(prefix + "_equity.csv", results[i].equity_curve);
        export_trades_csv(prefix + "_trades.csv", results[i].trades);
        results[i].surface_3d.export_csv(prefix + "_3d_surface.csv");
    }

    // --- 6. Export PnL surfaces for each strategy -------------------------
    std::cout << "[5/5] Generating PnL surface grids (IV × Spot → PnL)...\n";
    for (size_t i = 0; i < strategies.size(); ++i) {
        std::string sname = strategies[i].name;
        std::replace(sname.begin(), sname.end(), ' ', '_');
        std::string fname = "output/" + std::to_string(i) + "_" + sname + "_pnl_grid.csv";

        // Build a representative set of legs at current spot = 4500
        double S0 = 4500.0, r = 0.04, iv = 0.18, q = 0.015;
        auto legs = strategies[i].generate(S0, iv, r, q, VolRegime::Medium);
        if (legs.empty()) {
            // Try other regimes
            for (auto reg : {VolRegime::High, VolRegime::Low, VolRegime::Expansion, VolRegime::Crush}) {
                legs = strategies[i].generate(S0, iv, r, q, reg);
                if (!legs.empty()) break;
            }
        }
        if (!legs.empty()) {
            VolSurface3D::export_pnl_surface(fname, legs, r,
                                             strategies[i].dte_entry / DAYS_PER_YEAR, q,
                                             S0 * 0.85, S0 * 1.15, 60,
                                             0.08, 0.45, 40);
        }
    }

    // --- 7. Export time-evolving PnL surfaces (animated) -------------------
    std::cout << "[6/6] Generating time-evolving PnL surfaces (live animation data)...\n";
    {
        // Build a live spot + IV path from the MC median path
        // Simulate intraday resolution: interpolate between daily steps
        int intraday_steps_per_day = 60;  // ~1 point per 6.5 min (390 min / 60)
        int trade_horizon_days = 30;
        int total_intra_steps = trade_horizon_days * intraday_steps_per_day;

        // Use the median spot/iv from MC paths for the first 30 days
        std::vector<double> live_spot(total_intra_steps);
        std::vector<double> live_iv(total_intra_steps);
        std::vector<VolRegime> live_regime(total_intra_steps);

        // Extract daily medians for first 30 days
        std::vector<double> daily_spot(trade_horizon_days + 1);
        std::vector<double> daily_iv(trade_horizon_days + 1);
        for (int d = 0; d <= trade_horizon_days && d <= paths.n_steps; ++d) {
            std::vector<double> spots(paths.n_paths), vols(paths.n_paths);
            for (int p = 0; p < paths.n_paths; ++p) {
                spots[p] = paths.paths[p][d];
                vols[p]  = paths.vol_paths[p][d];
            }
            std::sort(spots.begin(), spots.end());
            std::sort(vols.begin(), vols.end());
            daily_spot[d] = spots[paths.n_paths / 2];
            daily_iv[d]   = vols[paths.n_paths / 2];
        }

        // Interpolate to intraday with micro-noise for realism
        std::mt19937 rng(123);
        std::normal_distribution<> noise(0.0, 1.0);
        std::vector<double> iv_hist;
        for (int t = 0; t < total_intra_steps; ++t) {
            double day_frac = static_cast<double>(t) / intraday_steps_per_day;
            int d0 = static_cast<int>(day_frac);
            int d1 = std::min(d0 + 1, trade_horizon_days);
            double alpha = day_frac - d0;

            double base_spot = daily_spot[d0] * (1.0 - alpha) + daily_spot[d1] * alpha;
            double base_iv   = daily_iv[d0] * (1.0 - alpha) + daily_iv[d1] * alpha;

            // Add intraday micro-movement
            double spot_noise = base_spot * 0.0003 * noise(rng);  // ~3bp noise
            double iv_noise   = base_iv * 0.002 * noise(rng);     // ~20bp vol noise

            live_spot[t] = base_spot + spot_noise;
            live_iv[t]   = std::max(base_iv + iv_noise, 0.05);

            double prev_iv = (t > 0) ? live_iv[t - 1] * 100.0 : live_iv[t] * 100.0;
            iv_hist.push_back(live_iv[t] * 100.0);
            auto rs = regime_clf.classify(live_iv[t] * 100.0, prev_iv, 0.0, iv_hist);
            live_regime[t] = rs.regime;
        }

        int n_frames = 120;  // 120 frames for smooth animation

        for (size_t i = 0; i < strategies.size(); ++i) {
            std::string sname = strategies[i].name;
            std::replace(sname.begin(), sname.end(), ' ', '_');
            std::string fname = "output/" + std::to_string(i) + "_" + sname + "_evolving.csv";

            double S0 = live_spot[0], r = 0.04, iv = live_iv[0], q = 0.015;
            auto legs = strategies[i].generate(S0, iv, r, q, VolRegime::Medium);
            if (legs.empty()) {
                for (auto reg : {VolRegime::High, VolRegime::Low, VolRegime::Expansion, VolRegime::Crush}) {
                    legs = strategies[i].generate(S0, iv, r, q, reg);
                    if (!legs.empty()) break;
                }
            }
            if (!legs.empty()) {
                VolSurface3D::export_evolving_surfaces(
                    fname, legs, r, q,
                    strategies[i].dte_entry / DAYS_PER_YEAR,
                    live_spot, live_iv, live_regime,
                    n_frames, 35, 25);
                std::cout << "   Exported: " << fname << " (" << n_frames << " frames)\n";
            }
        }
    }

    // --- 8. ARIMA-Enhanced Backtests -------------------------------------
    std::cout << "\n[7/7] Running ARIMA-enhanced backtests...\n";

    ARIMASignalGenerator arima_gen;
    arima_gen.lookback         = 60;
    arima_gen.forecast_horizon = 5;
    arima_gen.refit_freq       = 5;
    arima_gen.p_max            = 3;
    arima_gen.d_max            = 1;
    arima_gen.q_max            = 3;
    arima_gen.direction_threshold = 0.005;

    std::vector<Backtester::ARIMABacktestResult> arima_results;
    for (auto& strat : strategies) {
        auto res = Backtester::run_arima(paths, strat, bt_cfg, regime_clf, arima_gen);
        print_arima_report(strat.name, res);
        arima_results.push_back(std::move(res));
    }

    // Export ARIMA data
    std::cout << "   Exporting ARIMA forecast data...\n";
    for (size_t i = 0; i < strategies.size(); ++i) {
        std::string sname = strategies[i].name;
        std::replace(sname.begin(), sname.end(), ' ', '_');
        std::string prefix = "output/" + std::to_string(i) + "_" + sname;

        // ARIMA forecasts CSV
        export_arima_csv(prefix + "_arima_forecasts.csv", arima_results[i]);

        // ARIMA equity curve
        export_equity_csv(prefix + "_arima_equity.csv", arima_results[i].equity_curve);

        // ARIMA trade log
        {
            std::ofstream out(prefix + "_arima_trades.csv");
            out << "Strategy,DayOpened,DayClosed,SpotOpen,SpotClose,IVOpen,IVClose,"
                << "EntryCost,ExitValue,PnL,Regime,ARIMA_IV_Forecast,ARIMA_Direction,"
                << "ARIMA_Confidence,PredictedRegime,EarlyExit\n";
            for (auto& t : arima_results[i].trades) {
                out << t.strategy_name << "," << t.day_opened << "," << t.day_closed << ","
                    << std::fixed << std::setprecision(2)
                    << t.spot_at_open << "," << t.spot_at_close << ","
                    << t.iv_at_open << "," << t.iv_at_close << ","
                    << std::setprecision(4)
                    << t.entry_cost << "," << t.exit_value << ","
                    << std::setprecision(2) << t.pnl << ","
                    << VolRegimeClassifier::regime_name(t.regime_at_open) << ","
                    << std::setprecision(4) << t.arima_iv_forecast << ","
                    << t.arima_iv_direction << ","
                    << std::setprecision(4) << t.arima_confidence << ","
                    << VolRegimeClassifier::regime_name(t.predicted_regime) << ","
                    << (t.arima_early_exit ? "YES" : "NO") << "\n";
            }
        }

        // ARIMA model diagnostics
        if (arima_results[i].iv_model.n_obs > 0) {
            export_arima_diagnostics(prefix + "_arima_iv_diag.csv",
                                    arima_results[i].equity_curve.empty()
                                    ? std::vector<double>{}
                                    : [&]() {
                                        std::vector<double> ivs;
                                        for (auto& ep : arima_results[i].equity_curve)
                                            ivs.push_back(ep.iv);
                                        return ivs;
                                    }(),
                                    arima_results[i].iv_model, "IV");
        }
    }

    // --- Comparative summary (Standard vs ARIMA) -------------------------
    std::cout << "\n============================================================\n";
    std::cout << "  COMPARATIVE SUMMARY (Standard vs ARIMA-Enhanced)\n";
    std::cout << "============================================================\n";
    std::cout << std::left << std::setw(26) << "Strategy"
              << std::right << std::setw(10) << "Return"
              << std::setw(10) << "Sharpe"
              << std::setw(10) << "MaxDD"
              << std::setw(10) << "WinRate"
              << std::setw(10) << "Alpha"
              << "\n";
    std::cout << std::string(76, '-') << "\n";
    for (size_t i = 0; i < strategies.size(); ++i) {
        auto& s = results[i].stats;
        std::cout << std::left << std::setw(26) << strategies[i].name
                  << std::right << std::fixed << std::setprecision(2)
                  << std::setw(9) << s.annualised_return * 100.0 << "%"
                  << std::setw(10) << s.sharpe_ratio
                  << std::setw(9) << s.max_drawdown * 100.0 << "%"
                  << std::setw(9) << s.win_rate << "%"
                  << std::setw(9) << s.alpha * 100.0 << "%"
                  << "\n";
    }
    std::cout << std::string(76, '-') << "\n";
    std::cout << "  Benchmark (S&P 500):  "
              << results[0].stats.benchmark_return * 100.0 << "% total return\n";
    std::cout << "\n  ARIMA-Enhanced:\n";
    std::cout << std::string(76, '-') << "\n";
    std::cout << std::left << std::setw(26) << "Strategy (ARIMA)"
              << std::right << std::setw(10) << "Return"
              << std::setw(10) << "Sharpe"
              << std::setw(10) << "MaxDD"
              << std::setw(10) << "WinRate"
              << std::setw(10) << "Alpha"
              << "\n";
    std::cout << std::string(76, '-') << "\n";
    for (size_t i = 0; i < strategies.size(); ++i) {
        auto& s = arima_results[i].stats;
        std::cout << std::left << std::setw(26) << (strategies[i].name + " *")
                  << std::right << std::fixed << std::setprecision(2)
                  << std::setw(9) << s.annualised_return * 100.0 << "%"
                  << std::setw(10) << s.sharpe_ratio
                  << std::setw(9) << s.max_drawdown * 100.0 << "%"
                  << std::setw(9) << s.win_rate << "%"
                  << std::setw(9) << s.alpha * 100.0 << "%"
                  << "\n";
    }
    std::cout << std::string(76, '-') << "\n";
    std::cout << "  * = ARIMA-enhanced (predictive regime + IV direction filter + early exit)\n";
    std::cout << "============================================================\n";

    std::cout << "\nAll results exported to output/ directory.\n";
    std::cout << "Run 'python3 scripts/visualize.py' to generate 3D plots.\n";
    std::cout << "Run 'python3 scripts/animate_surfaces.py' for animated surfaces.\n";
    std::cout << "Run 'python3 scripts/arima_plots.py' for ARIMA diagnostics.\n\n";

    return 0;
}
