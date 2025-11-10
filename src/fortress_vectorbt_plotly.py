"""
VectorBT Strategy Comparison:
EMA Crossover vs RSI vs Buy & Hold Benchmark

Description:
    - Downloads price data from Yahoo Finance
    - Backtests EMA crossover & RSI strategies
    - Compares performance vs Buy & Hold benchmark
    - Displays interactive Plotly charts for equity and drawdowns
"""

import pandas as pd
import vectorbt as vbt
import plotly.graph_objects as go

# ─────────────────────────────────────────────
# ⚙️ CONFIGURATION
# ─────────────────────────────────────────────
TICKER = "SPY"
START_DATE = pd.Timestamp("2000-01-01")
INITIAL_CASH = 10_000
COMMISSION = 0.002
SHOW_PERFORMANCE_DETAILS = False
SHOW_INDIVIDUAL_STRATEGY_PLOTS = True
SAVE_RESULTS = True

# ─────────────────────────────────────────────
# 📥 DOWNLOAD DATA
# ─────────────────────────────────────────────
price = vbt.YFData.download(TICKER, start=START_DATE).get('Close')

# ─────────────────────────────────────────────
# 📈 STRATEGY 1: EMA CROSSOVER
# ─────────────────────────────────────────────
fast_ema = vbt.MA.run(price, window=50, ewm=True)
slow_ema = vbt.MA.run(price, window=200, ewm=True)

entries_ema = fast_ema.ma_crossed_above(slow_ema)
exits_ema = fast_ema.ma_crossed_below(slow_ema)

pf_ema = vbt.Portfolio.from_signals(
    price, entries_ema, exits_ema,
    fees=COMMISSION, init_cash=INITIAL_CASH, freq='1D'
)

# ─────────────────────────────────────────────
# 📊 STRATEGY 2: RSI STRATEGY
# ─────────────────────────────────────────────
rsi = vbt.RSI.run(price, window=14)
entries_rsi = rsi.rsi_crossed_above(30)
exits_rsi = rsi.rsi_crossed_below(70)

pf_rsi = vbt.Portfolio.from_signals(
    price, entries_rsi, exits_rsi,
    fees=COMMISSION, init_cash=INITIAL_CASH, freq='1D'
)

# ─────────────────────────────────────────────
# 💰 BENCHMARK (BUY & HOLD)
# ─────────────────────────────────────────────
pf_benchmark = vbt.Portfolio.from_holding(
    price, init_cash=INITIAL_CASH, freq='1D'
)

# ─────────────────────────────────────────────
# 🧾 PERFORMANCE SUMMARY
# ─────────────────────────────────────────────
if SHOW_PERFORMANCE_DETAILS:
    print("\n📈 EMA Crossover Strategy Stats:\n", pf_ema.stats())
    print("\n📊 RSI Strategy Stats:\n", pf_rsi.stats())
    print("\n💹 Benchmark Stats:\n", pf_benchmark.stats(
        metrics=["total_return", "max_dd", "sharpe_ratio", "sortino_ratio", "calmar_ratio"]
    ))

strategies = {
    "EMA Crossover": pf_ema,
    "RSI Strategy": pf_rsi,
    "Buy & Hold": pf_benchmark
}

metrics = ["total_return", "max_dd", "sharpe_ratio", "sortino_ratio", "calmar_ratio"]
summary = pd.DataFrame({name: pf.stats(metrics=metrics) for name, pf in strategies.items()})
print("\n📊 Strategy Comparison Summary:\n")
print(summary)

# ─────────────────────────────────────────────
# 📈 EQUITY CURVES (PLOTLY)
# ─────────────────────────────────────────────
fig_equity = go.Figure()
for name, pf in strategies.items():
    fig_equity.add_trace(
        go.Scatter(
            x=pf.value().index,
            y=pf.value().values,
            mode="lines",
            name=name
        )
    )

fig_equity.update_layout(
    title=f"{TICKER} Strategy Equity Curves (VectorBT)",
    xaxis_title="Date",
    yaxis_title="Portfolio Value ($)",
    template="plotly_dark",
    hovermode="x unified",
    legend=dict(x=0, y=1, traceorder="normal"),
)
fig_equity.show()

# ─────────────────────────────────────────────
# 📉 DRAWDOWNS (PLOTLY)
# ─────────────────────────────────────────────
fig_dd = go.Figure()
for name, pf in strategies.items():
    fig_dd.add_trace(
        go.Scatter(
            x=pf.drawdown().index,
            y=pf.drawdown().values,
            mode="lines",
            name=name
        )
    )

fig_dd.update_layout(
    title=f"{TICKER} Strategy Drawdowns (VectorBT)",
    xaxis_title="Date",
    yaxis_title="Drawdown (%)",
    template="plotly_dark",
    hovermode="x unified",
    legend=dict(x=0, y=1, traceorder="normal"),
)
fig_dd.show()

# ─────────────────────────────────────────────
# 📊 OPTIONAL: INDIVIDUAL STRATEGY PLOTS
# ─────────────────────────────────────────────
if SHOW_INDIVIDUAL_STRATEGY_PLOTS:
    pf_ema.plot(title="EMA Crossover").show()
    pf_rsi.plot(title="RSI Strategy").show()
 
# ─────────────────────────────────────────────
# 💾 OPTIONAL: SAVE RESULTS TO CSV
# ─────────────────────────────────────────────
if SAVE_RESULTS:
    summary.to_csv("strategy_summary.csv")
    print("✅ Strategy summary saved to strategy_summary.csv")
