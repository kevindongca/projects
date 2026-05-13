"""
Multi-Factor Quantitative Backtest
====================================
A factor-based equity backtest using momentum, reversal, and low volatility
signals to select and quarterly rebalance a 3-stock portfolio.
Benchmarked against SPY over the same period.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# ── Configuration ─────────────────────────────────────────────────────────────

TICKERS = "SHOP NVDA TSLA AMZN MSFT GOOGL META AXP BA WMT"
START_DATE = "2020-01-01"
END_DATE = "2026-01-01"
TOP_N = 3  # number of stocks to hold each quarter

# ── Data Download ──────────────────────────────────────────────────────────────

data = yf.download(TICKERS, start=START_DATE, end=END_DATE)
close = data["Close"]
returns = close.pct_change()

# ── Factor Signals ─────────────────────────────────────────────────────────────
# Momentum:  12-month return excluding the most recent month
# Reversal:  negative of 1-month return (mean reversion)
# Low Vol:   negative of 3-month rolling volatility (favour stable stocks)

momentum = close.shift(21) / close.shift(252) - 1
reversal = -(close / close.shift(21) - 1)
low_vol = -(returns.rolling(63).std())

# ── Signal Ranking & Composite Score ──────────────────────────────────────────
# Rank each signal cross-sectionally (across tickers) as percentile
# Average the three ranked signals into a single composite score

r1 = momentum.rank(axis=1, pct=True)
r2 = reversal.rank(axis=1, pct=True)
r3 = low_vol.rank(axis=1, pct=True)
composite = (r1 + r2 + r3) / 3
composite = composite.dropna()

# ── Quarterly Rebalance ────────────────────────────────────────────────────────
# At the start of each quarter, pick the top N stocks by composite score

holdings = {}
quarterly_dates = composite.resample("QE").first().index

for date in quarterly_dates:
    if date in composite.index:
        row = composite.loc[date]
        top_stocks = row.nlargest(TOP_N).index.tolist()
        holdings[date] = top_stocks

# ── Portfolio Simulation ───────────────────────────────────────────────────────
# Equal-weight the selected stocks each quarter and compute daily returns

portfolio_returns = []
dates = list(holdings.keys())

for i, (date, stocks) in enumerate(holdings.items()):
    start = date
    end = dates[i + 1] if i + 1 < len(dates) else composite.index[-1]

    period = returns.loc[start:end, stocks]
    period_return = period.mean(axis=1)
    portfolio_returns.append(period_return)

portfolio = pd.concat(portfolio_returns)
cumulative = (1 + portfolio).cumprod()

# ── Performance Metrics ────────────────────────────────────────────────────────

years = (cumulative.index[-1] - cumulative.index[0]).days / 365
cagr = (cumulative.iloc[-1]) ** (1 / years) - 1
sharpe = portfolio.mean() / portfolio.std() * np.sqrt(252)
max_drawdown = (cumulative / cumulative.cummax() - 1).min()

print(f"CAGR:         {cagr:.2%}")
print(f"Sharpe Ratio: {sharpe:.2f}")
print(f"Max Drawdown: {max_drawdown:.2%}")

# ── SPY Benchmark ──────────────────────────────────────────────────────────────

spy_data = yf.download("SPY", start=START_DATE, end=END_DATE)
spy_returns = spy_data["Close"].pct_change().dropna()
spy_cumulative = (1 + spy_returns).cumprod()

# ── Plot ───────────────────────────────────────────────────────────────────────

plt.figure(figsize=(12, 6))
plt.plot(cumulative, label="Portfolio")
plt.plot(spy_cumulative, label="SPY")
plt.title("Multi-Factor Portfolio vs SPY")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.tight_layout()
plt.show()
