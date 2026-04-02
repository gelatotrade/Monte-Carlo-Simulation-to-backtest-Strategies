# Monte Carlo Options Strategy Backtesting Engine

A high-performance **C++ backtesting engine** for complex options strategies with Monte Carlo simulation, volatility regime classification, and 3D visualization of P&L across implied volatility and S&P 500 levels.

---

## Visual Overview

### Strategy Equity Curves vs S&P 500 Benchmark
All six strategies compared against a passive S&P 500 buy-and-hold, with drawdown overlay:

![Equity Curves Comparison](docs/images/equity_curves_comparison.png)

### 3D Volatility Regime Map (IV × PnL × S&P 500)
Each point represents a daily snapshot, colour-coded by the active volatility regime:

| Long Butterfly | Short Iron Condor |
|:-:|:-:|
| ![Long Butterfly 3D](docs/images/2_Long_Butterfly_3d_regime.png) | ![Short Iron Condor 3D](docs/images/1_Short_Iron_Condor_3d_regime.png) |

| Vega Expansion Straddle | Short Iron Butterfly |
|:-:|:-:|
| ![Vega Expansion Straddle 3D](docs/images/0_Vega_Expansion_Straddle_3d_regime.png) | ![Short Iron Butterfly 3D](docs/images/3_Short_Iron_Butterfly_3d_regime.png) |

| Calendar Spread | Short Strangle |
|:-:|:-:|
| ![Calendar Spread 3D](docs/images/4_Calendar_Spread_3d_regime.png) | ![Short Strangle 3D](docs/images/5_Short_Strangle_3d_regime.png) |

### PnL Surface Plots (Spot × IV → PnL)
3D surfaces showing how each strategy's P&L changes across the full range of spot prices and implied volatility levels:

| Short Iron Condor | Long Butterfly |
|:-:|:-:|
| ![Iron Condor PnL Surface](docs/images/1_Short_Iron_Condor_pnl_surface.png) | ![Butterfly PnL Surface](docs/images/2_Long_Butterfly_pnl_surface.png) |

| Vega Expansion Straddle | Short Iron Butterfly |
|:-:|:-:|
| ![Straddle PnL Surface](docs/images/0_Vega_Expansion_Straddle_pnl_surface.png) | ![Iron Butterfly PnL Surface](docs/images/3_Short_Iron_Butterfly_pnl_surface.png) |

| Calendar Spread | Short Strangle |
|:-:|:-:|
| ![Calendar PnL Surface](docs/images/4_Calendar_Spread_pnl_surface.png) | ![Strangle PnL Surface](docs/images/5_Short_Strangle_pnl_surface.png) |

### Live Time-Evolving PnL Surfaces (Animated)

The surfaces below update **minute-by-minute** as spot price and implied volatility move along the simulated path. Watch how theta decay reshapes the surface, how the live cursor (red dot) tracks the current market position, and how the volatility regime shifts in real time:

#### All Strategies — Mid-Trade Snapshot
![Composite Snapshot](docs/images/evolving_surfaces_composite.png)

#### Short Iron Condor — Live Surface Animation
![Iron Condor Animated](docs/images/1_Short_Iron_Condor_animated.gif)

#### Long Butterfly — Live Surface Animation
![Butterfly Animated](docs/images/2_Long_Butterfly_animated.gif)

#### Vega Expansion Straddle — Live Surface Animation
![Straddle Animated](docs/images/0_Vega_Expansion_Straddle_animated.gif)

#### Short Iron Butterfly — Live Surface Animation
![Iron Butterfly Animated](docs/images/3_Short_Iron_Butterfly_animated.gif)

#### Calendar Spread — Live Surface Animation
![Calendar Animated](docs/images/4_Calendar_Spread_animated.gif)

#### Short Strangle — Live Surface Animation
![Strangle Animated](docs/images/5_Short_Strangle_animated.gif)

**What each animation shows:**
- **Surface** (Spot × IV → PnL): The 3D payoff landscape at the current time, colour-coded green (profit) to red (loss)
- **Red cursor**: Current live position (spot price × implied volatility) on the surface
- **Crosshairs**: Spot and IV slices through the current position
- **Title bar**: Real-time DTE, time (HH:MM), regime, spot, IV, and all Greeks (Delta, Gamma, Vega, Theta)
- **Regime indicator**: Coloured bar showing the active volatility regime (Low/Medium/High/Crush/Expansion)
- **Camera rotation**: Gentle azimuth wobble for full surface perspective

### Volatility Regime Analysis

| Regime Distribution & P&L Attribution | Implied Volatility Time-Series with Regime Shading |
|:-:|:-:|
| ![Regime Distribution](docs/images/regime_distribution.png) | ![IV Time-Series](docs/images/iv_timeseries_regimes.png) |

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Strategies Implemented](#strategies-implemented)
- [Volatility Regime Classification](#volatility-regime-classification)
- [3D Coordinate System & Live Evolving Surfaces](#3d-coordinate-system--live-evolving-surfaces)
- [ARIMA Forecasting Model](#arima-forecasting-model)
- [Monte Carlo Simulation](#monte-carlo-simulation)
- [Black-Scholes Pricing Engine](#black-scholes-pricing-engine)
- [Backtesting Engine](#backtesting-engine)
- [Performance Metrics](#performance-metrics)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Output Files](#output-files)
- [Configuration](#configuration)
- [Extending the Engine](#extending-the-engine)

---

## Overview

This engine combines **Monte Carlo path simulation** with a **volatility regime classifier** to backtest multi-leg options strategies on the S&P 500. It answers the fundamental question: *How does each options strategy perform across different implied volatility environments?*

### Key Features

- **Black-Scholes-Merton pricing** with full Greeks (Delta, Gamma, Vega, Theta, Rho)
- **Newton-Raphson implied volatility** solver
- **Stochastic volatility** Monte Carlo simulator (Heston-lite with CIR variance process)
- **6 options strategies**: Straddles, Strangles, Butterflies, Iron Condors, Iron Butterflies, Calendar Spreads
- **Volatility regime detection**: Low, Medium, High, Expansion, Crush
- **3D coordinate system** mapping IV × PnL × S&P 500 with regime colour-coding
- **Animated time-evolving PnL surfaces** — watch surfaces morph minute-by-minute as spot, IV, and DTE change
- **ARIMA(p,d,q) forecasting** — auto-fitted IV/spot models with predictive regime switching, early exit, and IV direction filters
- **S&P 500 buy-and-hold benchmark** comparison
- **Comprehensive statistics**: Sharpe, Sortino, max drawdown, win rate, profit factor, alpha
- **Per-regime P&L attribution** — see which vol environment each strategy profits in
- **CSV export** + **Python 3D visualization** suite

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        main.cpp                                  │
│  Orchestrates simulation → strategy config → backtest → export  │
└────────────┬──────────────┬──────────────┬──────────────────────┘
             │              │              │
    ┌────────▼────┐  ┌──────▼──────┐  ┌───▼────────────┐
    │ Monte Carlo │  │  Strategies │  │   Backtester   │
    │  Simulator  │  │   Factory   │  │    Engine      │
    │ (Heston)    │  │ (6 strats)  │  │ (trade logic)  │
    └────────┬────┘  └──────┬──────┘  └───┬──────┬─────┘
             │              │              │      │
    ┌────────▼──────────────▼──────────────▼──┐   │
    │              Black-Scholes Pricer        │   │
    │         (pricing, Greeks, implied vol)   │   │
    └─────────────────────┬───────────────────┘   │
                          │                       │
    ┌─────────────────────▼──────────────────┐    │
    │       Volatility Regime Classifier     │    │
    │  (Low / Medium / High / Crush / Exp)   │    │
    └─────────────────────┬──────────────────┘    │
                          │                       │
    ┌─────────────────────▼──────────────────┐    │
    │      ARIMA(p,d,q) Forecast Engine      │◄───┘
    │  Auto-fit · IV/Spot prediction · CI    │
    │  Predictive regime · Early exit signal  │
    └─────────────────────┬──────────────────┘
                          │
    ┌─────────────────────▼──────────────────┐
    │     3D Surface (IV × PnL × SP500)      │
    │  Static · Animated · ARIMA overlays    │
    │         CSV Export → Python viz         │
    └────────────────────────────────────────┘
```

---

## Strategies Implemented

### 1. Vega Expansion Straddle (Long)

**Thesis**: Buy ATM straddles when volatility is low or expanding — profit from increasing implied volatility (vega expansion).

```
Payoff:  Long Call (ATM) + Long Put (ATM)
         ╲        ╱
          ╲      ╱
           ╲    ╱
            ╲  ╱
             ╲╱  ← max loss = total premium
```

- **Entry condition**: Vol regime = Low or Expansion
- **DTE**: 30 days
- **Edge**: Long vega exposure benefits from IV increase

| 3D Regime Map | PnL Surface |
|:-:|:-:|
| ![Straddle 3D](docs/images/0_Vega_Expansion_Straddle_3d_regime.png) | ![Straddle Surface](docs/images/0_Vega_Expansion_Straddle_pnl_surface.png) |

### 2. Short Iron Condor

**Thesis**: Collect premium by selling OTM puts and calls with protective wings in medium-to-high vol environments.

```
Payoff:
    ─────╲                 ╱─────
          ╲_______________╱
          PL   PS   CS   CL

PL = Put Long, PS = Put Short, CS = Call Short, CL = Call Long
```

- **Entry condition**: Vol regime = Medium or High
- **DTE**: 30 days
- **Wing width**: 5% of spot per side
- **Edge**: Theta decay + mean-reversion of elevated IV

| 3D Regime Map | PnL Surface |
|:-:|:-:|
| ![Iron Condor 3D](docs/images/1_Short_Iron_Condor_3d_regime.png) | ![Iron Condor Surface](docs/images/1_Short_Iron_Condor_pnl_surface.png) |

### 3. Long Butterfly Spread

**Thesis**: Low-cost directional-neutral bet that the underlying pins near the current price. Profits from volatility crush.

```
Payoff:
              ╱╲
             ╱  ╲
            ╱    ╲
    ───────╱      ╲───────
          L    M    U
```

- **Entry condition**: Vol regime = Medium or Crush
- **DTE**: 21 days
- **Width**: 3% of spot
- **Edge**: Profits when spot stays range-bound and IV declines

| 3D Regime Map | PnL Surface |
|:-:|:-:|
| ![Butterfly 3D](docs/images/2_Long_Butterfly_3d_regime.png) | ![Butterfly Surface](docs/images/2_Long_Butterfly_pnl_surface.png) |

### 4. Short Iron Butterfly

**Thesis**: Maximum premium collection at ATM. Combines short straddle with protective wings for defined risk.

```
Payoff:
    ─────╲      ╱─────
          ╲    ╱
           ╲  ╱
            ╲╱
          L  M  U
```

- **Entry condition**: Vol regime = High or Crush
- **DTE**: 30 days
- **Edge**: Highest premium collection of all defined-risk strategies

| 3D Regime Map | PnL Surface |
|:-:|:-:|
| ![Iron Butterfly 3D](docs/images/3_Short_Iron_Butterfly_3d_regime.png) | ![Iron Butterfly Surface](docs/images/3_Short_Iron_Butterfly_pnl_surface.png) |

### 5. Calendar Spread (Horizontal)

**Thesis**: Exploit term-structure discrepancies. Short near-dated, long far-dated options at the same strike.

```
Time decay profile:
    Near-term (short): ████████░░  (fast decay)
    Far-term (long):   ██░░░░░░░░  (slow decay)
    Net theta:         Positive ✓
```

- **Entry condition**: Vol regime = Medium or Low
- **DTE**: Short 21-day, Long 60-day
- **Edge**: Profits from near-term theta decay while long-term option retains value

| 3D Regime Map | PnL Surface |
|:-:|:-:|
| ![Calendar 3D](docs/images/4_Calendar_Spread_3d_regime.png) | ![Calendar Surface](docs/images/4_Calendar_Spread_pnl_surface.png) |

### 6. Short Strangle

**Thesis**: Undefined-risk premium selling. Sell OTM calls and puts, profit from time decay in elevated vol.

```
Payoff:
    ╲                      ╱
     ╲____________________╱
      PK                 CK
```

- **Entry condition**: Vol regime = High
- **DTE**: 30 days
- **Width**: 5% OTM each side
- **Edge**: Maximum theta in high-IV, profits from vol mean-reversion

| 3D Regime Map | PnL Surface |
|:-:|:-:|
| ![Strangle 3D](docs/images/5_Short_Strangle_3d_regime.png) | ![Strangle Surface](docs/images/5_Short_Strangle_pnl_surface.png) |

---

## Volatility Regime Classification

The engine classifies each trading day into one of five regimes based on implied volatility dynamics:

| Regime | Condition | Trading Implication |
|--------|-----------|---------------------|
| **Low** | IV < low threshold | Buy vol (straddles, strangles) |
| **Medium** | Low < IV < high threshold | Balanced strategies (condors, butterflies) |
| **High** | IV > high threshold | Sell premium (iron condors, strangles) |
| **Expansion** | Monthly IV change > +expansion rate | Vol is spiking — long vega |
| **Crush** | Monthly IV change < −crush rate | Vol is collapsing — short vega |

### Classification Algorithm

```
1. Compute IV level (VIX proxy from stochastic vol simulation)
2. Compute daily IV change (dIV/dt)
3. Extrapolate to monthly rate (× 21 trading days)
4. Compute IV percentile rank over lookback window (252 days)
5. Compute IV − RV spread (implied vs realised)
6. Apply threshold-based classification
```

### Implied Volatility with Regime Shading

The IV time-series below shows how the regime classifier labels each trading day, with background shading by regime:

![IV Time-Series with Regimes](docs/images/iv_timeseries_regimes.png)

### Trade Distribution & P&L by Regime

![Regime Distribution](docs/images/regime_distribution.png)

### Configurable Parameters

```cpp
VolRegimeClassifier regime_clf;
regime_clf.low_iv_threshold  = 16.5;   // VIX below → "Low"
regime_clf.high_iv_threshold = 18.0;   // VIX above → "High"
regime_clf.expansion_rate    =  1.0;   // Monthly vol pts rise → "Expansion"
regime_clf.crush_rate        = -1.0;   // Monthly vol pts drop → "Crush"
```

---

## 3D Coordinate System & Live Evolving Surfaces

The engine generates data for a **3-dimensional coordinate system** where:

| Axis | Dimension | Description |
|------|-----------|-------------|
| **X** | Implied Volatility | VIX-like measure (annualised %) |
| **Y** | Strategy PnL | Cumulative profit/loss in USD |
| **Z** | S&P 500 Level | Underlying benchmark price |

Each point is colour-coded by the active **volatility regime** at that time step. This enables visual identification of:

- **Which IV levels generate the most P&L** for each strategy
- **How the S&P 500 level correlates** with strategy performance
- **Regime clustering** — where in the IV/Spot space each regime occurs
- **Risk pockets** — dangerous combinations of IV and spot for a given strategy

### PnL Surface (Static Grid)

The engine generates a **Spot × IV → PnL** surface grid for each strategy at entry, enabling:

- Visualization of the **full payoff landscape** across spot prices and implied volatility levels
- Identification of **breakeven boundaries** in IV-Spot space
- Understanding of **vega sensitivity** (how PnL shifts as IV changes)

### Time-Evolving Surfaces (Animated)

The key innovation: the engine exports **minute-by-minute surface snapshots** that show how each strategy's PnL surface **morphs in real time** as:

1. **Spot price moves** along the Monte Carlo simulated path (with intraday micro-noise)
2. **Implied volatility shifts** — the surface re-centres and reshapes
3. **Theta decay** progressively collapses the surface as DTE approaches zero
4. **Regime transitions** change the colour indicator and surface characteristics

![Evolving Surfaces Composite](docs/images/evolving_surfaces_composite.png)

Each animated frame contains:
- A **full Spot × IV → PnL** surface grid re-computed at the current DTE
- **Live Greeks** (Delta, Gamma, Vega, Theta) at the current spot/IV position
- A **red cursor** tracking the live market position on the surface
- **Crosshair slices** through the surface at the current spot and IV
- **Regime indicator** showing the active volatility environment
- **DTE countdown** and **intraday timestamp** (HH:MM)

The surface grid dynamically re-centres around the current spot (±10%) and IV (±50%) to maintain relevant resolution as the market moves.

```
Frame generation pipeline:
  1. Interpolate MC daily path → intraday resolution (60 steps/day)
  2. Add micro-noise (3bp spot, 20bp vol) for realistic tick-by-tick movement
  3. At each frame: recompute full Spot × IV grid at current remaining DTE
  4. Export: Spot, IV, PnL, Delta, Gamma, Vega, Theta per grid point
  5. Animate: 120 frames → compressed GIF (40 frames, 200ms/frame)
```

---

## ARIMA Forecasting Model

The engine includes a **full ARIMA(p,d,q) implementation from scratch in C++** — no external libraries required. It forecasts implied volatility and spot prices to enhance trading signals.

### What is ARIMA?

**A**uto**R**egressive **I**ntegrated **M**oving **A**verage — a time-series forecasting model with three components:

| Component | Formula | Purpose |
|-----------|---------|---------|
| **AR(p)** | y_t = c + Σ φ_i · y_{t-i} | Past values predict future values |
| **I(d)** | Δ^d y_t | Differencing to achieve stationarity |
| **MA(q)** | y_t += Σ θ_j · ε_{t-j} | Past forecast errors improve predictions |

### Auto-ARIMA Model Selection

The engine automatically selects the optimal (p, d, q) order via grid search:

```
1. Determine d: variance reduction test across d = 0, 1, 2
2. Grid search: p ∈ [0, p_max], q ∈ [0, q_max]
3. For each (p, d, q):
   a. Fit via Conditional Least Squares (iterative OLS)
   b. Compute AIC = −2·log(L) + 2·k
   c. Check AR stationarity (Σ|φ_i| < 1)
4. Select model with lowest AIC
```

### How ARIMA Enhances Trading Signals

The ARIMA model improves the backtester in three ways:

| Enhancement | Mechanism | Benefit |
|-------------|-----------|---------|
| **Predictive Regime** | Forecast IV → predict next regime before it arrives | Enter positions **ahead** of regime transitions |
| **IV Direction Filter** | Skip entries where ARIMA contradicts the strategy | Avoid selling vol before predicted spike |
| **Early Exit** | Close positions when ARIMA forecasts adverse regime | Cut losses before vol crush/expansion hits |

### ARIMA Diagnostics

| IV Forecast with 95% CI | Standard vs ARIMA Equity |
|:-:|:-:|
| ![IV Forecast](docs/images/2_Long_Butterfly_arima_forecast.png) | ![Equity Comparison](docs/images/2_Long_Butterfly_arima_equity_comparison.png) |

| ARIMA Signal Analysis | Model Diagnostics (Residual ACF) |
|:-:|:-:|
| ![Trade Analysis](docs/images/2_Long_Butterfly_arima_trade_analysis.png) | ![Diagnostics](docs/images/2_Long_Butterfly_arima_diagnostics.png) |

### Comparative Results: Standard vs ARIMA-Enhanced

![ARIMA Comparison](docs/images/arima_comparison.png)

### ARIMA Configuration

```cpp
ARIMASignalGenerator arima_gen;
arima_gen.lookback           = 60;     // rolling window for fitting
arima_gen.forecast_horizon   = 5;      // days ahead to forecast
arima_gen.refit_freq         = 5;      // re-estimate every 5 days
arima_gen.p_max              = 3;      // AR order search bound
arima_gen.d_max              = 1;      // differencing search bound
arima_gen.q_max              = 3;      // MA order search bound
arima_gen.direction_threshold = 0.005; // min % change for direction signal
```

### Implementation Details

The ARIMA engine (`include/arima.hpp`) includes:

- **OLS regression** via Gaussian elimination with partial pivoting
- **Conditional Least Squares** estimation (iterative AR + MA)
- **Confidence intervals** with horizon-dependent standard errors
- **Ljung-Box test** for residual autocorrelation diagnostics
- **Rolling forecasts** at configurable intervals during the backtest
- **CSV export** of forecasts, residuals, and fitted values

---

## Monte Carlo Simulation

The engine uses a **stochastic volatility model** (Heston-lite) with a CIR variance process:

### Price Dynamics (GBM with stochastic vol)

```
dS = (μ − q − ½v)dt + √v · √dt · Z₁
```

### Variance Dynamics (CIR process)

```
dv = κ(θ − v)dt + ξ√v · √dt · Z₂
```

### Correlation Structure

```
Z₂ = ρ · Z₁ + √(1 − ρ²) · Z_independent
```

### Default Parameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Initial spot | S₀ | 4500 | Approximate SPX level |
| Drift | μ | 8% | Expected annual return |
| Base volatility | σ | 18% | Starting annualised vol |
| Risk-free rate | r | 4% | Annualised |
| Dividend yield | q | 1.5% | Continuous |
| Paths | N | 10,000 | Monte Carlo paths |
| Steps | T | 504 | 2 years of trading days |
| Vol-of-vol | ξ | 35% | Volatility of variance |
| Long-run vol | θ | 18% | Mean-reversion target |
| Mean-reversion | κ | 3.0 | Speed of reversion |
| Spot-vol correlation | ρ | −0.70 | Leverage effect |

---

## Black-Scholes Pricing Engine

Full analytical Black-Scholes-Merton implementation:

### Pricing

```
C = S·e^{-qT}·N(d₁) − K·e^{-rT}·N(d₂)
P = K·e^{-rT}·N(−d₂) − S·e^{-qT}·N(−d₁)

d₁ = [ln(S/K) + (r − q + ½σ²)T] / (σ√T)
d₂ = d₁ − σ√T
```

### Greeks

| Greek | Formula | Sensitivity |
|-------|---------|-------------|
| **Delta** (Δ) | ∂C/∂S | Price sensitivity to spot |
| **Gamma** (Γ) | ∂²C/∂S² | Delta sensitivity to spot |
| **Vega** (ν) | ∂C/∂σ | Price sensitivity to IV (per 1%) |
| **Theta** (Θ) | ∂C/∂t | Time decay (per trading day) |
| **Rho** (ρ) | ∂C/∂r | Rate sensitivity (per 1%) |

### Implied Volatility

Newton-Raphson solver with:
- Convergence tolerance: 1e-8
- Maximum iterations: 100
- Floor: σ > 0.1%

---

## Backtesting Engine

### Trade Lifecycle

```
1. At each rebalance point (every N trading days):
   a. Classify current vol regime
   b. Query strategy signal generator
   c. If signal → open position (construct legs via BSM pricing)

2. During holding period:
   a. Track portfolio Greeks
   b. Monitor regime transitions

3. At expiry / target DTE:
   a. Re-price all legs at current spot & IV
   b. Compute P&L (mark-to-market − entry cost)
   c. Deduct transaction costs
   d. Record trade in log
```

### Portfolio Management

| Parameter | Default | Description |
|-----------|---------|-------------|
| Initial capital | $100,000 | Starting portfolio value |
| Position size | 5% | Capital allocation per trade |
| Rebalance frequency | 21 days | Monthly trade evaluation |
| Transaction cost | $0.65 | Per contract per leg |
| Contracts | 10 | Contracts per trade (100 shares each) |

---

## Performance Metrics

### Strategy Equity Curves with Drawdown

![Equity Curves](docs/images/equity_curves_comparison.png)

The engine computes the following for each strategy:

| Metric | Description |
|--------|-------------|
| **Total Return** | Cumulative P&L / initial capital |
| **Annualised Return** | Geometric annualisation |
| **Sharpe Ratio** | Risk-adjusted return (daily → annualised) |
| **Sortino Ratio** | Downside-deviation-adjusted return |
| **Max Drawdown** | Largest peak-to-trough decline |
| **Win Rate** | % of profitable trades |
| **Profit Factor** | Gross profit / gross loss |
| **Alpha** | Excess return over S&P 500 buy-and-hold |
| **Avg PnL / Trade** | Mean P&L across all trades |
| **Regime Attribution** | P&L and trade count per vol regime |

---

## Getting Started

### Prerequisites

- **C++17** compiler (GCC 9+, Clang 10+, MSVC 2019+)
- **CMake** 3.14+
- **Python 3.7+** with `matplotlib`, `numpy`, `pandas` (for visualization)

### Build & Run

```bash
# Clone the repository
git clone https://github.com/gelatotrade/Monte-Carlo-Simulation-to-backtest-Strategies.git
cd Monte-Carlo-Simulation-to-backtest-Strategies

# Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd ..

# Run the backtesting engine
./build/backtest

# Generate static 3D visualizations
pip install matplotlib numpy pandas Pillow
python3 scripts/visualize.py

# Generate animated time-evolving surface GIFs
python3 scripts/animate_surfaces.py
```

### Debug Build

```bash
mkdir build-debug && cd build-debug
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)
```

---

## Project Structure

```
Monte-Carlo-Simulation-to-backtest-Strategies/
├── CMakeLists.txt              # Build configuration
├── README.md                   # This file
│
├── docs/
│   └── images/                 # Visualization PNGs + GIFs (committed to repo)
│       ├── *_3d_regime.png     # 3D regime scatter plots
│       ├── *_pnl_surface.png   # PnL surface plots
│       ├── *_animated.gif      # Animated time-evolving PnL surfaces
│       ├── evolving_surfaces_composite.png
│       ├── *_arima_forecast.png      # ARIMA IV forecast fan charts
│       ├── *_arima_equity_comparison.png  # Standard vs ARIMA equity
│       ├── *_arima_trade_analysis.png # ARIMA signal analysis
│       ├── *_arima_diagnostics.png   # Residual ACF, fitted vs actual
│       ├── arima_comparison.png      # All strategies: Standard vs ARIMA
│       ├── equity_curves_comparison.png
│       ├── regime_distribution.png
│       └── iv_timeseries_regimes.png
│
├── include/                    # Header-only library
│   ├── common.hpp              # Types, constants, date helpers, normal CDF
│   ├── black_scholes.hpp       # BSM pricing engine & Greeks
│   ├── monte_carlo.hpp         # Stochastic vol MC simulator
│   ├── strategies.hpp          # Strategy factory (6 strategies) & evaluator
│   ├── vol_regime.hpp          # Vol regime classifier & 3D surface
│   ├── arima.hpp               # ARIMA(p,d,q) model, auto-fit, forecast, diagnostics
│   └── backtester.hpp          # Backtesting engine with full statistics
│
├── src/
│   └── main.cpp                # Entry point: config → simulate → backtest → export
│
├── scripts/
│   ├── visualize.py            # Python 3D visualization suite (5 plot types)
│   ├── animate_surfaces.py     # Animated time-evolving surface GIF generator
│   └── arima_plots.py          # ARIMA forecast diagnostics & comparison plots
│
├── output/                     # Generated at runtime
│   ├── *_equity.csv            # Equity curves per strategy
│   ├── *_trades.csv            # Trade logs per strategy
│   ├── *_3d_surface.csv        # 3D scatter data (IV × PnL × SP500)
│   ├── *_pnl_grid.csv          # PnL surface grids (Spot × IV)
│   ├── *_evolving.csv          # Time-evolving surface data (120 frames)
│   ├── *_arima_forecasts.csv   # Rolling ARIMA IV/spot forecasts
│   ├── *_arima_equity.csv      # ARIMA-enhanced equity curves
│   ├── *_arima_trades.csv      # Trade log with ARIMA metadata
│   ├── *_arima_iv_diag.csv     # ARIMA model residuals & diagnostics
│   └── plots/                  # Generated PNG/GIF visualizations
│       ├── *_3d_regime.png     # 3D regime scatter plots
│       ├── *_pnl_surface.png   # PnL surface plots
│       ├── *_animated.gif      # Animated evolving surface GIFs
│       ├── equity_curves_comparison.png
│       ├── regime_distribution.png
│       └── iv_timeseries_regimes.png
│
└── build/                      # CMake build directory
```

---

## Output Files

### CSV Exports

| File Pattern | Contents | Use |
|-------------|----------|-----|
| `*_equity.csv` | Day, Equity, Benchmark, Drawdown, IV, Regime | Equity curve plotting |
| `*_trades.csv` | Full trade log with entry/exit prices, IV, regime | Trade analysis |
| `*_3d_surface.csv` | IV, PnL, SP500, Regime per time step | 3D scatter visualisation |
| `*_pnl_grid.csv` | Spot × IV → PnL grid | 3D surface plot |
| `*_evolving.csv` | Frame, Minute, DTE, LiveSpot, LiveIV, Regime, grid points with Greeks | Animated surface GIFs |

### Visualizations

| Plot | Description |
|------|-------------|
| **3D Regime Scatter** | IV × PnL × SP500 colour-coded by vol regime |
| **PnL Surface** | 3D surface: how PnL changes across spot & IV space |
| **Animated Surface GIFs** | Time-evolving surfaces with live cursor, Greeks, regime, DTE countdown |
| **Composite Snapshot** | All 6 strategies' live surfaces at mid-trade point |
| **Equity Curves** | All strategies vs S&P 500 benchmark with drawdown |
| **Regime Distribution** | Pie chart (trade count) + bar chart (PnL by regime) |
| **IV Time-Series** | Implied vol over time with regime-shaded background |

---

## Configuration

All parameters are configurable in `src/main.cpp`:

### Monte Carlo Parameters

```cpp
MonteCarlo::Config mc_cfg;
mc_cfg.S0        = 4500.0;    // initial SPX level
mc_cfg.mu        = 0.08;      // expected drift
mc_cfg.sigma     = 0.18;      // base volatility
mc_cfg.n_paths   = 10000;     // simulation paths
mc_cfg.n_steps   = 504;       // trading days (2 years)
mc_cfg.vol_of_vol = 0.35;     // stochastic vol parameter
mc_cfg.rho        = -0.70;    // leverage effect
```

### Backtest Parameters

```cpp
Backtester::Config bt_cfg;
bt_cfg.initial_capital  = 100000.0;
bt_cfg.rebalance_freq   = 21;        // monthly
bt_cfg.transaction_cost = 0.65;       // per contract
bt_cfg.contracts        = 10;
```

### Regime Thresholds

```cpp
VolRegimeClassifier regime_clf;
regime_clf.low_iv_threshold  = 16.5;
regime_clf.high_iv_threshold = 18.0;
regime_clf.expansion_rate    =  1.0;
regime_clf.crush_rate        = -1.0;
```

---

## Extending the Engine

### Adding a New Strategy

1. Create a `StrategySignal` factory function in `main.cpp`:

```cpp
StrategySignal make_my_strategy() {
    StrategySignal sig;
    sig.name      = "My Custom Strategy";
    sig.hold_days = 30;
    sig.dte_entry = 30.0;
    sig.generate  = [](double S, double iv, double r, double q,
                       VolRegime regime) -> std::vector<Leg> {
        if (regime == VolRegime::High) {
            // Build legs using StrategyFactory helpers or manually
            return StrategyFactory::straddle(S, 30.0/252.0, S, r, iv, q, false);
        }
        return {};  // empty = no trade signal
    };
    return sig;
}
```

2. Add to the strategies vector:

```cpp
strategies.push_back(make_my_strategy());
```

### Adding a New Leg Type

Use the `Leg` struct directly:

```cpp
Leg my_leg;
my_leg.type        = OptionType::Call;
my_leg.strike      = 4600.0;
my_leg.expiry_T    = 30.0 / 252.0;
my_leg.quantity     = -1;  // short
my_leg.entry_price = BlackScholes::calculate(OptionType::Call, S, 4600.0, T, r, sigma).price;
```

### Custom Regime Logic

Subclass or modify `VolRegimeClassifier::classify()` to add conditions based on:
- IV term structure (contango / backwardation)
- IV skew (put-call skew)
- Realized-to-implied vol ratio
- VIX-of-VIX (VVIX) level
- Macro indicators

---

## Mathematical References

- **Black, F. & Scholes, M.** (1973). "The Pricing of Options and Corporate Liabilities." *Journal of Political Economy*.
- **Heston, S.L.** (1993). "A Closed-Form Solution for Options with Stochastic Volatility." *Review of Financial Studies*.
- **Cox, J.C., Ingersoll, J.E. & Ross, S.A.** (1985). "A Theory of the Term Structure of Interest Rates." *Econometrica*.
- **Natenberg, S.** (2015). *Option Volatility and Pricing*. McGraw-Hill.

---

## License

This project is provided for educational and research purposes. Use at your own risk. Past simulated performance does not guarantee future results.
