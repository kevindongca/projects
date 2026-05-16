"""
Multi-Factor Quantitative Backtest — Modular Version with Dashboard
====================================================================
Lessons 15-17: Transaction costs, risk metrics, professional dashboard
Refactored into modular functions for readability and reusability.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ── Configuration ──────────────────────────────────────────────────────────────

TICKERS = "AAPL NVDA MSFT GOOGL META TSLA AMD CRM JPM GS AXP V MA AMZN WMT COST NKE JNJ UNH PFE XOM CVX BA CAT HON SHOP"
START_DATE = '2020-01-01'
END_DATE = '2026-01-01'
TOP_N = 3
COST_PER_TRADE = 0.001  # 0.1% per trade
NUM_TRADES = 3          # buying 3 new stocks each quarter
TRAIN_YEARS = 2
TEST_MONTHS = 6

# ── Data Download ──────────────────────────────────────────────────────────────

def download_data(tickers, start, end):
    """Download price data and compute daily returns."""
    data = yf.download(tickers, start=start, end=end)
    close = data['Close']
    returns = close.pct_change()
    return close, returns

# ── Factor Signals ─────────────────────────────────────────────────────────────

def compute_signals(close, returns):
    """Compute all 4 factor signals from price data."""
    momentum = close.shift(21) / close.shift(252) - 1
    reversal = -(close / close.shift(21) - 1)
    low_vol = -(returns.rolling(63).std())
    high_52w = close / close.rolling(252).max()
    return momentum, reversal, low_vol, high_52w

def rank_signals(momentum, reversal, low_vol, high_52w):
    """Percentile rank each signal cross-sectionally."""
    r1 = momentum.rank(axis=1, pct=True)
    r2 = reversal.rank(axis=1, pct=True)
    r3 = low_vol.rank(axis=1, pct=True)
    r4 = high_52w.rank(axis=1, pct=True)
    return r1, r2, r3, r4

def build_composite(weights, r1, r2, r3, r4):
    """Build weighted composite score from ranked signals."""
    w1, w2, w3, w4 = weights
    return (w1 * r1 + w2 * r2 + w3 * r3 + w4 * r4).dropna()

# ── Portfolio Simulation ───────────────────────────────────────────────────────

def run_portfolio(comp, returns, cost_per_trade=0, num_trades=0):
    """
    Run quarterly rebalance portfolio simulation.
    Returns a Series of daily portfolio returns.
    """
    holdings = {}
    quarterly_dates = comp.resample('QE').first().index
    for date in quarterly_dates:
        if date in comp.index:
            holdings[date] = comp.loc[date].nlargest(TOP_N).index.tolist()

    if not holdings:
        return None

    portfolio_returns = []
    dates = list(holdings.keys())
    for i, (date, stocks) in enumerate(holdings.items()):
        start = date
        end = dates[i + 1] if i + 1 < len(dates) else comp.index[-1]
        period_return = returns.loc[start:end, stocks].mean(axis=1)
        period_return.iloc[0] -= cost_per_trade * num_trades  # apply transaction cost
        portfolio_returns.append(period_return)

    return pd.concat(portfolio_returns)

# ── Sharpe Objective for Optimization ─────────────────────────────────────────

def get_sharpe(weights, r1, r2, r3, r4, returns):
    """Negative Sharpe ratio for scipy minimization."""
    comp = build_composite(weights, r1, r2, r3, r4)
    portfolio = run_portfolio(comp, returns)
    if portfolio is None or portfolio.std() == 0:
        return 0
    sharpe = portfolio.mean() / portfolio.std() * np.sqrt(252)
    return -sharpe

# ── Weight Optimization ────────────────────────────────────────────────────────

def optimize_weights(r1, r2, r3, r4, returns):
    """Find optimal factor weights using scipy SLSQP."""
    result = minimize(
        get_sharpe,
        [0.25, 0.25, 0.25, 0.25],
        args=(r1, r2, r3, r4, returns),
        method='SLSQP',
        bounds=[(0, 1)] * 4,
        constraints={'type': 'eq', 'fun': lambda w: sum(w) - 1}
    )
    return result.x

# ── Risk Metrics ───────────────────────────────────────────────────────────────

def compute_metrics(portfolio):
    """Compute full risk metric suite from daily return series."""
    cumulative = (1 + portfolio).cumprod()
    years = (cumulative.index[-1] - cumulative.index[0]).days / 365
    cagr = (cumulative.iloc[-1]) ** (1 / years) - 1
    sharpe = portfolio.mean() / portfolio.std() * np.sqrt(252)
    drawdown = (cumulative / cumulative.cummax() - 1).min()
    calmar = cagr / abs(drawdown)
    sortino = portfolio.mean() / portfolio[portfolio < 0].std() * np.sqrt(252)
    var_95 = portfolio.quantile(0.05)
    return cumulative, cagr, sharpe, sortino, calmar, drawdown, var_95

# ── Dashboard ──────────────────────────────────────────────────────────────────

def plot_dashboard(cumulative, portfolio, spy_cumulative):
    """Plot 3-panel professional dashboard."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle('Multi-Factor Portfolio Dashboard', fontsize=16, fontweight='bold')

    # Panel 1 — Cumulative Returns
    axes[0].plot(cumulative, label='Portfolio', color='blue')
    axes[0].plot(spy_cumulative, label='SPY', color='orange')
    axes[0].set_title('Cumulative Returns')
    axes[0].set_ylabel('Growth of $1')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel 2 — Drawdown
    drawdown_series = (cumulative / cumulative.cummax() - 1)
    axes[1].fill_between(drawdown_series.index, drawdown_series, 0,
                         color='red', alpha=0.4)
    axes[1].set_title('Drawdown')
    axes[1].set_ylabel('Drawdown %')
    axes[1].grid(True, alpha=0.3)

    # Panel 3 — Rolling Sharpe
    rolling_sharpe = portfolio.rolling(63).mean() / portfolio.rolling(63).std() * np.sqrt(252)
    axes[2].plot(rolling_sharpe, color='green')
    axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[2].set_title('Rolling Sharpe Ratio (63 days)')
    axes[2].set_ylabel('Sharpe')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # Download data
    close, returns = download_data(TICKERS, START_DATE, END_DATE)

    # Compute and rank signals
    momentum, reversal, low_vol, high_52w = compute_signals(close, returns)
    r1, r2, r3, r4 = rank_signals(momentum, reversal, low_vol, high_52w)

    # Walk forward testing
    train_start = pd.Timestamp(START_DATE)
    all_test_returns = []

    while True:
        train_end = train_start + pd.DateOffset(years=TRAIN_YEARS)
        test_end = train_end + pd.DateOffset(months=TEST_MONTHS)

        if test_end > pd.Timestamp(END_DATE):
            break

        # Slice to training period
        r1_train, r2_train = r1.loc[train_start:train_end], r2.loc[train_start:train_end]
        r3_train, r4_train = r3.loc[train_start:train_end], r4.loc[train_start:train_end]
        returns_train = returns.loc[train_start:train_end]

        # Optimize weights on training data
        weights = optimize_weights(r1_train, r2_train, r3_train, r4_train, returns_train)
        print(f"Period {train_end.date()} — weights: {weights.round(2)}")

        # Apply to test period
        r1_test, r2_test = r1.loc[train_end:test_end], r2.loc[train_end:test_end]
        r3_test, r4_test = r3.loc[train_end:test_end], r4.loc[train_end:test_end]
        returns_test = returns.loc[train_end:test_end]

        comp_test = build_composite(weights, r1_test, r2_test, r3_test, r4_test)
        test_portfolio = run_portfolio(comp_test, returns_test, COST_PER_TRADE, NUM_TRADES)

        if test_portfolio is not None:
            all_test_returns.append(test_portfolio)

        train_start += pd.DateOffset(months=TEST_MONTHS)

    # Compute metrics
    portfolio = pd.concat(all_test_returns)
    cumulative, cagr, sharpe, sortino, calmar, drawdown, var_95 = compute_metrics(portfolio)

    print(f"\nWalk Forward Results:")
    print(f"CAGR:         {cagr:.2%}")
    print(f"Sharpe:       {sharpe:.2f}")
    print(f"Sortino:      {sortino:.2f}")
    print(f"Calmar:       {calmar:.2f}")
    print(f"Max Drawdown: {drawdown:.2%}")
    print(f"VaR (95%):    {var_95:.2%}")

    # SPY benchmark
    spy_data = yf.download('SPY', start=START_DATE, end=END_DATE)
    spy_returns = spy_data['Close'].pct_change().dropna()
    spy_cumulative = (1 + spy_returns).cumprod()

    # Plot dashboard
    plot_dashboard(cumulative, portfolio, spy_cumulative)