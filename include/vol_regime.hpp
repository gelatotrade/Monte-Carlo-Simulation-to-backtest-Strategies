#pragma once
#include "common.hpp"

namespace opts {

// ---------------------------------------------------------------------------
// Volatility Regime Classifier
// Classifies market conditions into regimes based on IV level, IV change,
// and realized-vs-implied spread.
// ---------------------------------------------------------------------------
class VolRegimeClassifier {
public:
    struct RegimeState {
        VolRegime regime       = VolRegime::Medium;
        double    iv_level     = 0.0;   // absolute IV (VIX proxy)
        double    iv_change    = 0.0;   // dIV / dt (annualised)
        double    iv_rv_spread = 0.0;   // IV − RV
        double    iv_percentile = 0.0;  // rank of current IV in lookback
    };

    // Thresholds (configurable)
    double low_iv_threshold     = 14.0;
    double high_iv_threshold    = 24.0;
    double expansion_rate       =  3.0;   // vol points per month rise
    double crush_rate           = -3.0;

    int    percentile_lookback  = 252;     // trading days

    // Classify a single snapshot given history
    RegimeState classify(double current_iv, double prev_iv,
                         double realised_vol,
                         const std::vector<double>& iv_history) const {
        RegimeState s;
        s.iv_level     = current_iv;
        s.iv_change    = (current_iv - prev_iv);  // daily change in vol pts
        s.iv_rv_spread = current_iv - realised_vol * 100.0;

        // IV percentile rank
        if (!iv_history.empty()) {
            int lookback = std::min(percentile_lookback, static_cast<int>(iv_history.size()));
            int count = 0;
            for (int i = static_cast<int>(iv_history.size()) - lookback;
                 i < static_cast<int>(iv_history.size()); ++i) {
                if (iv_history[i] <= current_iv) ++count;
            }
            s.iv_percentile = 100.0 * count / lookback;
        }

        // Monthly change rate (approx 21 trading days)
        double monthly_change = s.iv_change * 21.0;

        if (monthly_change >= expansion_rate) {
            s.regime = VolRegime::Expansion;
        } else if (monthly_change <= crush_rate) {
            s.regime = VolRegime::Crush;
        } else if (current_iv < low_iv_threshold) {
            s.regime = VolRegime::Low;
        } else if (current_iv > high_iv_threshold) {
            s.regime = VolRegime::High;
        } else {
            s.regime = VolRegime::Medium;
        }
        return s;
    }

    static std::string regime_name(VolRegime r) {
        switch (r) {
            case VolRegime::Low:       return "Low";
            case VolRegime::Medium:    return "Medium";
            case VolRegime::High:      return "High";
            case VolRegime::Crush:     return "Crush";
            case VolRegime::Expansion: return "Expansion";
        }
        return "Unknown";
    }
};

// ---------------------------------------------------------------------------
// 3-D Coordinate System for IV / PnL / Time visualisation
// Stores points that can be exported for 3-D plotting
// ---------------------------------------------------------------------------
struct Point3D {
    double x = 0.0;  // dimension 1 (e.g. IV level)
    double y = 0.0;  // dimension 2 (e.g. PnL)
    double z = 0.0;  // dimension 3 (e.g. time / DTE / spot)
    VolRegime regime = VolRegime::Medium;
    std::string label;
};

class VolSurface3D {
public:
    std::vector<Point3D> points;

    void add(double x, double y, double z, VolRegime regime = VolRegime::Medium,
             const std::string& label = "") {
        points.push_back({x, y, z, regime, label});
    }

    // Export as CSV for external 3-D plotting (Python / gnuplot / etc.)
    void export_csv(const std::string& filename,
                    const std::string& x_label = "IV",
                    const std::string& y_label = "PnL",
                    const std::string& z_label = "SP500") const {
        std::ofstream out(filename);
        out << x_label << "," << y_label << "," << z_label
            << ",Regime,Label\n";
        for (auto& p : points) {
            out << p.x << "," << p.y << "," << p.z << ","
                << VolRegimeClassifier::regime_name(p.regime) << ","
                << p.label << "\n";
        }
    }

    // Export grid for surface plot (IV x Spot → PnL)
    static void export_pnl_surface(const std::string& filename,
                                   const std::vector<Leg>& legs,
                                   double r, double T, double q,
                                   double spot_min, double spot_max, int spot_steps,
                                   double iv_min,   double iv_max,   int iv_steps) {
        std::ofstream out(filename);
        out << "Spot,IV,PnL\n";
        for (int si = 0; si <= spot_steps; ++si) {
            double S = spot_min + (spot_max - spot_min) * si / spot_steps;
            for (int vi = 0; vi <= iv_steps; ++vi) {
                double iv = iv_min + (iv_max - iv_min) * vi / iv_steps;
                auto pnl = StrategyFactory::evaluate(legs, S, r, iv, T, q);
                out << S << "," << iv << "," << pnl.net_pnl << "\n";
            }
        }
    }

    // -----------------------------------------------------------------------
    // Export time-evolving PnL surfaces — simulates how the surface changes
    // minute-by-minute as spot and IV move along a live path.
    //
    // Each frame = one time snapshot with:
    //   - A full Spot × IV → PnL grid (the surface at that instant)
    //   - Current live spot & IV (the "cursor" on the surface)
    //   - Remaining DTE (theta decay reshapes the surface)
    //   - Active vol regime
    // -----------------------------------------------------------------------
    static void export_evolving_surfaces(
            const std::string& filename,
            const std::vector<Leg>& legs,
            double r, double q,
            double initial_T,                          // DTE at entry (years)
            const std::vector<double>& spot_path,      // live spot over time
            const std::vector<double>& iv_path,        // live IV over time (decimal)
            const std::vector<VolRegime>& regime_path, // regime at each step
            int n_frames,                              // how many frames to export
            int spot_grid_steps = 40,
            int iv_grid_steps   = 30) {

        if (spot_path.empty() || iv_path.empty()) return;

        int total_steps = static_cast<int>(spot_path.size());
        int frame_skip  = std::max(1, total_steps / n_frames);

        std::ofstream out(filename);
        out << "Frame,Minute,DTE_days,LiveSpot,LiveIV,Regime,"
            << "Spot,IV,PnL,Delta,Gamma,Vega,Theta\n";

        for (int frame = 0; frame < n_frames && frame * frame_skip < total_steps; ++frame) {
            int t = frame * frame_skip;
            double live_spot = spot_path[t];
            double live_iv   = iv_path[t];
            VolRegime regime = (t < static_cast<int>(regime_path.size()))
                               ? regime_path[t] : VolRegime::Medium;

            // Time remaining shrinks each frame
            double frac_elapsed = static_cast<double>(t) / total_steps;
            double T_remaining  = initial_T * (1.0 - frac_elapsed);
            double dte_days     = T_remaining * DAYS_PER_YEAR;

            // Grid centered on live spot, range ±10%
            double spot_min = live_spot * 0.90;
            double spot_max = live_spot * 1.10;
            // IV range centered on live IV, ±50% relative
            double iv_min = std::max(live_iv * 0.50, 0.05);
            double iv_max = live_iv * 1.50;

            int minute = static_cast<int>(frac_elapsed * initial_T * DAYS_PER_YEAR * 390); // 390 min/day

            for (int si = 0; si <= spot_grid_steps; ++si) {
                double S = spot_min + (spot_max - spot_min) * si / spot_grid_steps;
                for (int vi = 0; vi <= iv_grid_steps; ++vi) {
                    double iv = iv_min + (iv_max - iv_min) * vi / iv_grid_steps;
                    auto res = StrategyFactory::evaluate(legs, S, r, iv,
                                                         std::max(T_remaining, 0.0), q);
                    out << frame << "," << minute << ","
                        << std::fixed << std::setprecision(2) << dte_days << ","
                        << std::setprecision(2) << live_spot << ","
                        << std::setprecision(4) << live_iv << ","
                        << VolRegimeClassifier::regime_name(regime) << ","
                        << std::setprecision(2) << S << ","
                        << std::setprecision(4) << iv << ","
                        << std::setprecision(4) << res.net_pnl << ","
                        << std::setprecision(4) << res.delta << ","
                        << std::setprecision(6) << res.gamma << ","
                        << std::setprecision(4) << res.vega << ","
                        << std::setprecision(4) << res.theta << "\n";
                }
            }
        }
    }
};

} // namespace opts
