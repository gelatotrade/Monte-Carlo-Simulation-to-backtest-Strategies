#pragma once

#include <cmath>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <numeric>
#include <random>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cassert>
#include <functional>
#include <memory>
#include <stdexcept>
#include <array>

namespace opts {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
constexpr double PI        = 3.14159265358979323846;
constexpr double SQRT_2PI  = 2.50662827463100050242;
constexpr double DAYS_PER_YEAR = 252.0;   // trading days

// ---------------------------------------------------------------------------
// Enumerations
// ---------------------------------------------------------------------------
enum class OptionType  { Call, Put };
enum class VolRegime   { Low, Medium, High, Crush, Expansion };

// ---------------------------------------------------------------------------
// Utility – cumulative normal distribution (Abramowitz & Stegun approx.)
// ---------------------------------------------------------------------------
inline double norm_cdf(double x) {
    static constexpr double a1 =  0.254829592;
    static constexpr double a2 = -0.284496736;
    static constexpr double a3 =  1.421413741;
    static constexpr double a4 = -1.453152027;
    static constexpr double a5 =  1.061405429;
    static constexpr double p  =  0.3275911;

    int sign = (x < 0) ? -1 : 1;
    x = std::fabs(x) / std::sqrt(2.0);
    double t = 1.0 / (1.0 + p * x);
    double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * std::exp(-x * x);
    return 0.5 * (1.0 + sign * y);
}

inline double norm_pdf(double x) {
    return std::exp(-0.5 * x * x) / SQRT_2PI;
}

// ---------------------------------------------------------------------------
// Date helpers (simple YYYY-MM-DD)
// ---------------------------------------------------------------------------
struct Date {
    int year = 0, month = 0, day = 0;

    Date() = default;
    Date(int y, int m, int d) : year(y), month(m), day(d) {}

    static Date from_string(const std::string& s) {
        Date d;
        char dash;
        std::istringstream ss(s);
        ss >> d.year >> dash >> d.month >> dash >> d.day;
        return d;
    }

    std::string to_string() const {
        std::ostringstream ss;
        ss << year << "-"
           << std::setw(2) << std::setfill('0') << month << "-"
           << std::setw(2) << std::setfill('0') << day;
        return ss.str();
    }

    // Naive day-count (sufficient for backtesting granularity)
    int to_serial() const {
        int y = year, m = month;
        if (m <= 2) { y--; m += 12; }
        return 365 * y + y / 4 - y / 100 + y / 400 + (153 * (m - 3) + 2) / 5 + day - 1;
    }

    double year_frac_to(const Date& other) const {
        return static_cast<double>(other.to_serial() - to_serial()) / 365.0;
    }

    bool operator<(const Date& o)  const { return to_serial() < o.to_serial(); }
    bool operator<=(const Date& o) const { return to_serial() <= o.to_serial(); }
    bool operator==(const Date& o) const { return to_serial() == o.to_serial(); }
    bool operator!=(const Date& o) const { return !(*this == o); }
};

// ---------------------------------------------------------------------------
// Market snapshot at a point in time
// ---------------------------------------------------------------------------
struct MarketData {
    Date   date;
    double spot         = 0.0;   // underlying price (S&P 500)
    double risk_free    = 0.04;  // annualised risk-free rate
    double realised_vol = 0.0;   // trailing realised vol
    double vix          = 0.0;   // VIX (implied vol proxy)
    double div_yield    = 0.0;   // continuous dividend yield
};

} // namespace opts
