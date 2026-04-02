#pragma once
#include "common.hpp"
#include <numeric>
#include <cassert>
#include <cmath>
#include <stdexcept>

namespace opts {

// =========================================================================
// ARIMA(p,d,q) Model — Pure C++ Implementation
//
// Autoregressive Integrated Moving Average for time-series forecasting.
// Used here to forecast implied volatility and spot returns, enabling
// predictive regime transitions and smarter entry/exit signals.
//
// Components:
//   AR(p) — autoregressive:  y_t = c + Σ φ_i · y_{t-i}
//   I(d)  — differencing:    Δ^d y_t  (stationarity transform)
//   MA(q) — moving average:  y_t += Σ θ_j · ε_{t-j}
//
// Estimation: Conditional Least Squares (CSS) for AR, innovations for MA
// Model selection: AIC / BIC grid search over (p,d,q) candidates
// =========================================================================

// ---------------------------------------------------------------------------
// Matrix utilities for OLS regression (small matrices only)
// ---------------------------------------------------------------------------
namespace linalg {

using Mat = std::vector<std::vector<double>>;
using Vec = std::vector<double>;

inline Mat transpose(const Mat& A) {
    if (A.empty()) return {};
    int m = static_cast<int>(A.size()), n = static_cast<int>(A[0].size());
    Mat T(n, Vec(m));
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            T[j][i] = A[i][j];
    return T;
}

inline Mat multiply(const Mat& A, const Mat& B) {
    int m = static_cast<int>(A.size());
    int k = static_cast<int>(B.size());
    int n = static_cast<int>(B[0].size());
    Mat C(m, Vec(n, 0.0));
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            for (int l = 0; l < k; ++l)
                C[i][j] += A[i][l] * B[l][j];
    return C;
}

inline Vec multiply_vec(const Mat& A, const Vec& x) {
    int m = static_cast<int>(A.size());
    int n = static_cast<int>(x.size());
    Vec y(m, 0.0);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            y[i] += A[i][j] * x[j];
    return y;
}

// Solve Ax = b via Gaussian elimination with partial pivoting
inline Vec solve(Mat A, Vec b) {
    int n = static_cast<int>(A.size());
    for (int col = 0; col < n; ++col) {
        // Partial pivoting
        int max_row = col;
        for (int row = col + 1; row < n; ++row)
            if (std::abs(A[row][col]) > std::abs(A[max_row][col]))
                max_row = row;
        std::swap(A[col], A[max_row]);
        std::swap(b[col], b[max_row]);

        if (std::abs(A[col][col]) < 1e-14) continue; // singular

        for (int row = col + 1; row < n; ++row) {
            double factor = A[row][col] / A[col][col];
            for (int j = col; j < n; ++j)
                A[row][j] -= factor * A[col][j];
            b[row] -= factor * b[col];
        }
    }
    // Back substitution
    Vec x(n, 0.0);
    for (int i = n - 1; i >= 0; --i) {
        x[i] = b[i];
        for (int j = i + 1; j < n; ++j)
            x[i] -= A[i][j] * x[j];
        if (std::abs(A[i][i]) > 1e-14)
            x[i] /= A[i][i];
    }
    return x;
}

// OLS: solve (X^T X) β = X^T y
inline Vec ols(const Mat& X, const Vec& y) {
    Mat Xt = transpose(X);
    Mat XtX = multiply(Xt, X);
    Vec Xty = multiply_vec(Xt, y);
    return solve(XtX, Xty);
}

} // namespace linalg

// ---------------------------------------------------------------------------
// Differencing / Integration helpers
// ---------------------------------------------------------------------------
inline std::vector<double> difference(const std::vector<double>& series, int d) {
    std::vector<double> result = series;
    for (int i = 0; i < d; ++i) {
        std::vector<double> diff(result.size() - 1);
        for (size_t j = 0; j < diff.size(); ++j)
            diff[j] = result[j + 1] - result[j];
        result = std::move(diff);
    }
    return result;
}

inline std::vector<double> integrate(const std::vector<double>& diff_series,
                                     const std::vector<double>& original, int d) {
    if (d == 0) return diff_series;
    // Undo one level of differencing using the last original value
    std::vector<double> result(diff_series.size() + 1);
    // We need the last d values from the original before the forecast window
    result[0] = original.back();
    for (size_t i = 0; i < diff_series.size(); ++i)
        result[i + 1] = result[i] + diff_series[i];
    // Remove the seed value, keep only forecasts
    std::vector<double> forecasts(result.begin() + 1, result.end());
    if (d > 1) {
        // Recursively undo remaining differencing levels
        return integrate(forecasts, original, d - 1);
    }
    return forecasts;
}

// ---------------------------------------------------------------------------
// ARIMA Model
// ---------------------------------------------------------------------------
struct ARIMAParams {
    int p = 1;   // AR order
    int d = 1;   // differencing order
    int q = 1;   // MA order

    double intercept = 0.0;
    std::vector<double> ar_coeffs;   // φ_1, ..., φ_p
    std::vector<double> ma_coeffs;   // θ_1, ..., θ_q
    double sigma2 = 0.0;            // residual variance

    // Model selection criteria
    double aic = 0.0;
    double bic = 0.0;
    double rmse = 0.0;

    int n_obs = 0;   // number of observations used for fitting
};

class ARIMA {
public:
    // -----------------------------------------------------------------
    // Fit ARIMA(p,d,q) to a time series
    // -----------------------------------------------------------------
    static ARIMAParams fit(const std::vector<double>& series, int p, int d, int q) {
        ARIMAParams params;
        params.p = p;
        params.d = d;
        params.q = q;

        // Step 1: Apply differencing
        std::vector<double> y = difference(series, d);
        int n = static_cast<int>(y.size());
        int start = std::max(p, q);

        if (n <= start + 1) {
            // Not enough data
            params.ar_coeffs.assign(p, 0.0);
            params.ma_coeffs.assign(q, 0.0);
            return params;
        }

        params.n_obs = n - start;

        // Step 2: Estimate AR parameters via OLS
        // Then iteratively estimate MA via innovation residuals
        params.ar_coeffs.resize(p, 0.0);
        params.ma_coeffs.resize(q, 0.0);

        // Iterative estimation (CSS approach)
        std::vector<double> residuals(n, 0.0);
        int max_iter = 25;

        for (int iter = 0; iter < max_iter; ++iter) {
            // Build regression matrix: y_t ~ c + Σ φ_i·y_{t-i} + Σ θ_j·ε_{t-j}
            int n_vars = 1 + p + q;  // intercept + AR + MA
            linalg::Mat X(params.n_obs, linalg::Vec(n_vars, 0.0));
            linalg::Vec Y(params.n_obs);

            for (int t = start; t < n; ++t) {
                int row = t - start;
                Y[row] = y[t];
                X[row][0] = 1.0;  // intercept

                // AR terms
                for (int i = 0; i < p; ++i)
                    X[row][1 + i] = y[t - 1 - i];

                // MA terms (use current residual estimates)
                for (int j = 0; j < q; ++j) {
                    int idx = t - 1 - j;
                    X[row][1 + p + j] = (idx >= 0) ? residuals[idx] : 0.0;
                }
            }

            // OLS estimation
            auto beta = linalg::ols(X, Y);

            params.intercept = beta[0];
            for (int i = 0; i < p; ++i) params.ar_coeffs[i] = beta[1 + i];
            for (int j = 0; j < q; ++j) params.ma_coeffs[j] = beta[1 + p + j];

            // Update residuals
            double old_sse = 0.0;
            for (int t = 0; t < n; ++t) {
                double pred = params.intercept;
                for (int i = 0; i < p; ++i) {
                    if (t - 1 - i >= 0) pred += params.ar_coeffs[i] * y[t - 1 - i];
                }
                for (int j = 0; j < q; ++j) {
                    int idx = t - 1 - j;
                    if (idx >= 0) pred += params.ma_coeffs[j] * residuals[idx];
                }
                residuals[t] = y[t] - pred;
                if (t >= start) old_sse += residuals[t] * residuals[t];
            }

            params.sigma2 = old_sse / params.n_obs;
        }

        // Compute AIC / BIC
        int k = 1 + p + q;  // number of parameters
        double log_lik = -0.5 * params.n_obs * (std::log(2.0 * PI) + std::log(params.sigma2) + 1.0);
        params.aic = -2.0 * log_lik + 2.0 * k;
        params.bic = -2.0 * log_lik + std::log(static_cast<double>(params.n_obs)) * k;
        params.rmse = std::sqrt(params.sigma2);

        return params;
    }

    // -----------------------------------------------------------------
    // Forecast h steps ahead
    // Returns: vector of h forecasted values (in original scale if d > 0)
    // -----------------------------------------------------------------
    static std::vector<double> forecast(const std::vector<double>& series,
                                        const ARIMAParams& params, int h) {
        std::vector<double> y = difference(series, params.d);
        int n = static_cast<int>(y.size());

        // Compute residuals on the fitted data
        std::vector<double> residuals(n, 0.0);
        for (int t = 0; t < n; ++t) {
            double pred = params.intercept;
            for (int i = 0; i < params.p; ++i) {
                if (t - 1 - i >= 0) pred += params.ar_coeffs[i] * y[t - 1 - i];
            }
            for (int j = 0; j < params.q; ++j) {
                int idx = t - 1 - j;
                if (idx >= 0) pred += params.ma_coeffs[j] * residuals[idx];
            }
            residuals[t] = y[t] - pred;
        }

        // Extend y and residuals for forecasting
        std::vector<double> y_ext = y;
        std::vector<double> res_ext = residuals;
        std::vector<double> forecasts_diff(h);

        for (int t = 0; t < h; ++t) {
            double pred = params.intercept;
            int cur = n + t;

            // AR terms (can use past forecasts)
            for (int i = 0; i < params.p; ++i) {
                int idx = cur - 1 - i;
                if (idx >= 0 && idx < static_cast<int>(y_ext.size()))
                    pred += params.ar_coeffs[i] * y_ext[idx];
            }

            // MA terms (future residuals = 0, past residuals from fit)
            for (int j = 0; j < params.q; ++j) {
                int idx = cur - 1 - j;
                if (idx >= 0 && idx < static_cast<int>(res_ext.size()))
                    pred += params.ma_coeffs[j] * res_ext[idx];
            }

            forecasts_diff[t] = pred;
            y_ext.push_back(pred);
            res_ext.push_back(0.0);  // future shocks = 0 (expectation)
        }

        // Undo differencing
        if (params.d > 0) {
            return integrate(forecasts_diff, series, params.d);
        }
        return forecasts_diff;
    }

    // -----------------------------------------------------------------
    // Confidence intervals for forecasts
    // Returns: pair of (lower, upper) vectors at given confidence level
    // -----------------------------------------------------------------
    static std::pair<std::vector<double>, std::vector<double>>
    forecast_ci(const std::vector<double>& series,
                const ARIMAParams& params, int h, double z = 1.96) {
        auto fc = forecast(series, params, h);
        double se = std::sqrt(params.sigma2);

        std::vector<double> lower(h), upper(h);
        for (int t = 0; t < h; ++t) {
            // SE grows with sqrt(t+1) for random walk component
            double se_t = se * std::sqrt(static_cast<double>(t + 1));
            if (params.d == 0) se_t = se;  // stationary: constant SE
            lower[t] = fc[t] - z * se_t;
            upper[t] = fc[t] + z * se_t;
        }
        return {lower, upper};
    }

    // -----------------------------------------------------------------
    // Auto-ARIMA: Grid search over (p,d,q) to minimise AIC
    //   p_max, d_max, q_max: upper bounds for search
    // -----------------------------------------------------------------
    static ARIMAParams auto_fit(const std::vector<double>& series,
                                int p_max = 5, int d_max = 2, int q_max = 5) {
        ARIMAParams best;
        best.aic = 1e18;

        // Determine optimal d via augmented Dickey-Fuller-like check
        int best_d = 0;
        for (int d = 0; d <= d_max; ++d) {
            auto diff = difference(series, d);
            if (diff.size() < 10) continue;

            // Simple stationarity test: check if mean of differences
            // is significantly different from zero relative to variance
            double mean = 0.0;
            for (auto v : diff) mean += v;
            mean /= diff.size();
            double var = 0.0;
            for (auto v : diff) var += (v - mean) * (v - mean);
            var /= diff.size();

            // If variance is small relative to original, we're stationary enough
            double orig_var = 0.0;
            double orig_mean = 0.0;
            for (auto v : series) orig_mean += v;
            orig_mean /= series.size();
            for (auto v : series) orig_var += (v - orig_mean) * (v - orig_mean);
            orig_var /= series.size();

            if (var < orig_var * 0.5 || d == 0) {
                best_d = d;
                if (var < orig_var * 0.3) break;  // well-stationary
            }
        }

        // Grid search over p and q
        for (int d = std::max(0, best_d - 1); d <= std::min(best_d + 1, d_max); ++d) {
            for (int p = 0; p <= p_max; ++p) {
                for (int q = 0; q <= q_max; ++q) {
                    if (p == 0 && q == 0) continue;  // need at least one

                    auto diff = difference(series, d);
                    if (static_cast<int>(diff.size()) <= std::max(p, q) + 2)
                        continue;

                    auto params = fit(series, p, d, q);

                    // Check for NaN or unreasonable values
                    if (std::isnan(params.aic) || std::isinf(params.aic)) continue;

                    // Check AR stationarity (all roots outside unit circle)
                    bool stable = true;
                    double ar_sum = 0.0;
                    for (auto c : params.ar_coeffs) {
                        ar_sum += std::abs(c);
                    }
                    if (ar_sum >= 1.0) stable = false;

                    if (stable && params.aic < best.aic) {
                        best = params;
                    }
                }
            }
        }

        return best;
    }

    // -----------------------------------------------------------------
    // Compute residuals for diagnostic analysis
    // -----------------------------------------------------------------
    static std::vector<double> residuals(const std::vector<double>& series,
                                         const ARIMAParams& params) {
        auto y = difference(series, params.d);
        int n = static_cast<int>(y.size());
        std::vector<double> res(n, 0.0);

        for (int t = 0; t < n; ++t) {
            double pred = params.intercept;
            for (int i = 0; i < params.p; ++i) {
                if (t - 1 - i >= 0) pred += params.ar_coeffs[i] * y[t - 1 - i];
            }
            for (int j = 0; j < params.q; ++j) {
                int idx = t - 1 - j;
                if (idx >= 0) pred += params.ma_coeffs[j] * res[idx];
            }
            res[t] = y[t] - pred;
        }
        return res;
    }

    // -----------------------------------------------------------------
    // Ljung-Box test for residual autocorrelation
    // Returns test statistic Q (compare against chi-squared(lags - p - q))
    // -----------------------------------------------------------------
    static double ljung_box(const std::vector<double>& residuals, int lags = 20) {
        int n = static_cast<int>(residuals.size());
        if (n < lags + 1) return 0.0;

        double mean = 0.0;
        for (auto r : residuals) mean += r;
        mean /= n;

        double var = 0.0;
        for (auto r : residuals) var += (r - mean) * (r - mean);

        double Q = 0.0;
        for (int k = 1; k <= lags; ++k) {
            double acf_k = 0.0;
            for (int t = k; t < n; ++t)
                acf_k += (residuals[t] - mean) * (residuals[t - k] - mean);
            acf_k /= var;
            Q += (acf_k * acf_k) / (n - k);
        }
        Q *= n * (n + 2);
        return Q;
    }
};

// ---------------------------------------------------------------------------
// ARIMA Forecast State — wraps forecast + CI for use in backtesting
// ---------------------------------------------------------------------------
struct ARIMAForecast {
    double point_forecast = 0.0;    // best estimate
    double lower_95       = 0.0;    // 95% CI lower bound
    double upper_95       = 0.0;    // 95% CI upper bound
    double change_pct     = 0.0;    // predicted % change from current
    int    direction      = 0;      // -1 = declining, 0 = flat, +1 = rising

    // Confidence in direction (0-1)
    double confidence     = 0.0;
};

// ---------------------------------------------------------------------------
// ARIMA Signal Generator — runs rolling ARIMA on IV series
// to produce forecast-based trading signals
// ---------------------------------------------------------------------------
class ARIMASignalGenerator {
public:
    int    lookback       = 60;     // rolling window for fitting
    int    forecast_horizon = 5;    // days ahead to forecast
    int    refit_freq     = 5;      // re-estimate every N days
    int    p_max          = 3;      // auto-ARIMA search bounds
    int    d_max          = 1;
    int    q_max          = 3;
    double direction_threshold = 0.005;  // min % change to declare direction

    struct ForecastResult {
        ARIMAForecast iv_forecast;
        ARIMAForecast spot_forecast;
        ARIMAParams   iv_model;
        ARIMAParams   spot_model;
        VolRegime     predicted_regime = VolRegime::Medium;
    };

    // Generate forecast at time t given history up to t
    ForecastResult generate(const std::vector<double>& iv_history,
                            const std::vector<double>& spot_history,
                            int t,
                            double low_iv_thresh = 16.5,
                            double high_iv_thresh = 18.0) const {
        ForecastResult result;

        // Extract rolling windows
        int start = std::max(0, t - lookback);
        int end   = t + 1;

        if (end - start < 15) return result;  // not enough data

        std::vector<double> iv_window(iv_history.begin() + start,
                                      iv_history.begin() + std::min(end, static_cast<int>(iv_history.size())));
        std::vector<double> spot_window(spot_history.begin() + start,
                                        spot_history.begin() + std::min(end, static_cast<int>(spot_history.size())));

        // Fit ARIMA on IV
        result.iv_model = ARIMA::auto_fit(iv_window, p_max, d_max, q_max);
        auto iv_fc = ARIMA::forecast(iv_window, result.iv_model, forecast_horizon);
        auto [iv_lo, iv_hi] = ARIMA::forecast_ci(iv_window, result.iv_model, forecast_horizon);

        if (!iv_fc.empty()) {
            double current_iv = iv_window.back();
            double pred_iv    = iv_fc.back();  // terminal forecast
            result.iv_forecast.point_forecast = pred_iv;
            result.iv_forecast.lower_95 = iv_lo.back();
            result.iv_forecast.upper_95 = iv_hi.back();
            result.iv_forecast.change_pct = (pred_iv - current_iv) / current_iv;

            if (result.iv_forecast.change_pct > direction_threshold)
                result.iv_forecast.direction = 1;
            else if (result.iv_forecast.change_pct < -direction_threshold)
                result.iv_forecast.direction = -1;
            else
                result.iv_forecast.direction = 0;

            // Confidence: how much of CI is on one side of current level
            double range = result.iv_forecast.upper_95 - result.iv_forecast.lower_95;
            if (range > 1e-10) {
                double dist_from_current = std::abs(pred_iv - current_iv);
                result.iv_forecast.confidence = std::min(1.0, 2.0 * dist_from_current / range);
            }
        }

        // Fit ARIMA on spot (log returns for stationarity)
        if (spot_window.size() > 15) {
            std::vector<double> log_spots(spot_window.size());
            for (size_t i = 0; i < spot_window.size(); ++i)
                log_spots[i] = std::log(spot_window[i]);

            result.spot_model = ARIMA::auto_fit(log_spots, p_max, 1, q_max);
            auto spot_fc = ARIMA::forecast(log_spots, result.spot_model, forecast_horizon);
            auto [sp_lo, sp_hi] = ARIMA::forecast_ci(log_spots, result.spot_model, forecast_horizon);

            if (!spot_fc.empty()) {
                double current_spot = spot_window.back();
                double pred_spot    = std::exp(spot_fc.back());
                result.spot_forecast.point_forecast = pred_spot;
                result.spot_forecast.lower_95 = std::exp(sp_lo.back());
                result.spot_forecast.upper_95 = std::exp(sp_hi.back());
                result.spot_forecast.change_pct = (pred_spot - current_spot) / current_spot;

                if (result.spot_forecast.change_pct > direction_threshold)
                    result.spot_forecast.direction = 1;
                else if (result.spot_forecast.change_pct < -direction_threshold)
                    result.spot_forecast.direction = -1;
                else
                    result.spot_forecast.direction = 0;
            }
        }

        // Predict regime from IV forecast
        if (!iv_fc.empty()) {
            double pred_iv = result.iv_forecast.point_forecast;
            double current_iv = iv_window.back();
            double monthly_change = (pred_iv - current_iv) / forecast_horizon * 21.0;

            if (monthly_change > 1.0)
                result.predicted_regime = VolRegime::Expansion;
            else if (monthly_change < -1.0)
                result.predicted_regime = VolRegime::Crush;
            else if (pred_iv < low_iv_thresh)
                result.predicted_regime = VolRegime::Low;
            else if (pred_iv > high_iv_thresh)
                result.predicted_regime = VolRegime::High;
            else
                result.predicted_regime = VolRegime::Medium;
        }

        return result;
    }
};

// ---------------------------------------------------------------------------
// Export ARIMA diagnostics to CSV
// ---------------------------------------------------------------------------
inline void export_arima_diagnostics(const std::string& filename,
                                     const std::vector<double>& series,
                                     const ARIMAParams& params,
                                     const std::string& series_name = "IV") {
    std::ofstream out(filename);
    out << "t," << series_name << ",Fitted,Residual,Forecast_5d,Lower_95,Upper_95\n";

    auto y = difference(series, params.d);
    auto res = ARIMA::residuals(series, params);
    int n = static_cast<int>(y.size());
    int start = std::max(params.p, params.q);

    // Fitted values
    for (int t = start; t < n; ++t) {
        double fitted = y[t] - res[t];
        out << t << "," << y[t] << "," << fitted << "," << res[t] << ",,\n";
    }

    // Forecasts
    auto fc = ARIMA::forecast(series, params, 10);
    auto [lo, hi] = ARIMA::forecast_ci(series, params, 10);
    for (int t = 0; t < static_cast<int>(fc.size()); ++t) {
        out << (n + t) << ",," << fc[t] << ",," << fc[t] << ","
            << lo[t] << "," << hi[t] << "\n";
    }
}

} // namespace opts
