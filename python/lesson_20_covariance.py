"""
Multi-Factor Quantitative Backtest — Covariance Risk Decomposition
==================================================================
Lesson 20: Uses PCA loadings to build a diversified portfolio.
Compares Factor Model vs PCA-Diversified Model vs SPY.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns

# ── Configuration ──────────────────────────────────────────────────────────────

TICKERS = "AAPL NVDA MSFT GOOGL META TSLA AMD CRM JPM GS AXP V MA AMZN WMT COST NKE JNJ UNH PFE XOM CVX BA CAT HON SHOP"
START_DATE = '2020-01-01'
END_DATE = '2026-01-01'
TOP_N = 3
COST_PER_TRADE = 0.001
NUM_TRADES = 3
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

# ── PCA Analysis ───────────────────────────────────────────────────────────────

def run_pca_analysis(returns, n_components=5):
    """Decompose return matrix into principal components."""
    clean_returns = returns.dropna()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(clean_returns)
    pca = PCA(n_components=n_components)
    pca.fit(scaled)
    explained = pca.explained_variance_ratio_
    cumulative_explained = np.cumsum(explained)
    print("\nPCA Results:")
    for i, (var, cum) in enumerate(zip(explained, cumulative_explained)):
        print(f"PC{i+1}: {var:.2%} variance explained (cumulative: {cum:.2%})")
    loadings = pd.DataFrame(
        pca.components_.T,
        index=returns.columns,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    return pca, explained, loadings

# ── Diversified Stock Selection ────────────────────────────────────────────────

def diversified_stock_selection(comp_row, loadings, top_n=3):
    """
    Select stocks diversified across PC2 factor exposure.

    Instead of just top 3 by composite score:
    - Pick top 1 from positive PC2 stocks (value/energy)
    - Pick top 1 from negative PC2 stocks (tech/growth)
    - Pick top 1 overall by composite score

    This ensures portfolio isn't concentrated in one factor regime.
    """
    pc2 = loadings['PC2']

    # split universe into positive and negative PC2 stocks
    positive_pc2 = pc2[pc2 > 0].index.tolist()
    negative_pc2 = pc2[pc2 < 0].index.tolist()

    # filter composite scores to each group
    comp_positive = comp_row[comp_row.index.isin(positive_pc2)]
    comp_negative = comp_row[comp_row.index.isin(negative_pc2)]

    selected = []

    # best stock from positive PC2 group (value/energy)
    if len(comp_positive) > 0:
        selected.append(comp_positive.idxmax())

    # best stock from negative PC2 group (tech/growth)
    if len(comp_negative) > 0:
        selected.append(comp_negative.idxmax())

    # fill remaining slots with highest overall composite
    remaining = comp_row[~comp_row.index.isin(selected)]
    while len(selected) < top_n and len(remaining) > 0:
        best = remaining.idxmax()
        selected.append(best)
        remaining = remaining.drop(best)

    return selected[:top_n]

# ── Portfolio Simulation ───────────────────────────────────────────────────────

def run_portfolio(comp, returns, cost_per_trade=0, num_trades=0):
    """Run quarterly rebalance portfolio simulation from composite score."""
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
        period_return.iloc[0] -= cost_per_trade * num_trades
        portfolio_returns.append(period_return)

    return pd.concat(portfolio_returns)

# ── Risk Metrics ───────────────────────────────────────────────────────────────

def compute_metrics(portfolio):
    """Compute full risk metric suite."""
    cumulative = (1 + portfolio).cumprod()
    years = (cumulative.index[-1] - cumulative.index[0]).days / 365
    cagr = (cumulative.iloc[-1]) ** (1 / years) - 1
    sharpe = portfolio.mean() / portfolio.std() * np.sqrt(252)
    drawdown = (cumulative / cumulative.cummax() - 1).min()
    calmar = cagr / abs(drawdown)
    sortino = portfolio.mean() / portfolio[portfolio < 0].std() * np.sqrt(252)
    var_95 = portfolio.quantile(0.05)
    return cumulative, cagr, sharpe, sortino, calmar, drawdown, var_95

# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # Download data
    close, returns = download_data(TICKERS, START_DATE, END_DATE)

    # Compute and rank signals
    momentum, reversal, low_vol, high_52w = compute_signals(close, returns)
    r1, r2, r3, r4 = rank_signals(momentum, reversal, low_vol, high_52w)

    # Run PCA on full return history to get loadings
    print("Running PCA...")
    pca, explained, loadings = run_pca_analysis(returns)
    print("\nTop stocks by PC2 loading (value/energy):")
    print(loadings['PC2'].sort_values(ascending=False).head(5))
    print("\nBottom stocks by PC2 loading (tech/growth):")
    print(loadings['PC2'].sort_values(ascending=True).head(5))

    # Walk forward testing — factor model vs diversified model
    train_start = pd.Timestamp(START_DATE)
    all_factor_returns = []
    all_div_returns = []

    while True:
        train_end = train_start + pd.DateOffset(years=TRAIN_YEARS)
        test_end = train_end + pd.DateOffset(months=TEST_MONTHS)

        if test_end > pd.Timestamp(END_DATE):
            break

        # Slice signals to test period
        r1_test = r1.loc[train_end:test_end]
        r2_test = r2.loc[train_end:test_end]
        r3_test = r3.loc[train_end:test_end]
        r4_test = r4.loc[train_end:test_end]
        returns_test = returns.loc[train_end:test_end]

        comp_test = build_composite([0.25, 0.25, 0.25, 0.25],
                                     r1_test, r2_test, r3_test, r4_test)

        # ── Factor model (top 3 by composite score) ───────────────────────────
        factor_portfolio = run_portfolio(comp_test, returns_test,
                                         COST_PER_TRADE, NUM_TRADES)
        if factor_portfolio is not None:
            all_factor_returns.append(factor_portfolio)

        # ── Diversified model (PCA-aware selection) ───────────────────────────
        quarterly_dates = comp_test.resample('QE').first().index
        div_period_returns = []
        div_dates = [d for d in quarterly_dates if d in comp_test.index]

        for i, date in enumerate(div_dates):
            comp_row = comp_test.loc[date]
            div_stocks = diversified_stock_selection(comp_row, loadings, top_n=TOP_N)
            print(f"Period {date.date()} — Diversified picks: {div_stocks}")

            start = date
            end = div_dates[i + 1] if i + 1 < len(div_dates) else comp_test.index[-1]

            period_return = returns_test.loc[start:end, div_stocks].mean(axis=1)
            period_return.iloc[0] -= COST_PER_TRADE * NUM_TRADES
            div_period_returns.append(period_return)

        if div_period_returns:
            all_div_returns.append(pd.concat(div_period_returns))

        train_start += pd.DateOffset(months=TEST_MONTHS)

    # ── Factor portfolio metrics ───────────────────────────────────────────────
    factor_portfolio = pd.concat(all_factor_returns)
    factor_cumulative, cagr, sharpe, sortino, calmar, drawdown, var_95 = \
        compute_metrics(factor_portfolio)

    print(f"\nFactor Model Results:")
    print(f"CAGR:         {cagr:.2%}")
    print(f"Sharpe:       {sharpe:.2f}")
    print(f"Sortino:      {sortino:.2f}")
    print(f"Calmar:       {calmar:.2f}")
    print(f"Max Drawdown: {drawdown:.2%}")
    print(f"VaR (95%):    {var_95:.2%}")

    # ── Diversified portfolio metrics ──────────────────────────────────────────
    div_portfolio = pd.concat(all_div_returns)
    div_cumulative, cagr, sharpe, sortino, calmar, drawdown, var_95 = \
        compute_metrics(div_portfolio)

    print(f"\nDiversified Model Results:")
    print(f"CAGR:         {cagr:.2%}")
    print(f"Sharpe:       {sharpe:.2f}")
    print(f"Sortino:      {sortino:.2f}")
    print(f"Calmar:       {calmar:.2f}")
    print(f"Max Drawdown: {drawdown:.2%}")
    print(f"VaR (95%):    {var_95:.2%}")

    # ── SPY benchmark ──────────────────────────────────────────────────────────
    spy_data = yf.download('SPY', start=START_DATE, end=END_DATE)
    spy_returns = spy_data['Close'].pct_change().dropna()
    spy_cumulative = (1 + spy_returns).cumprod()

    # ── Dashboard ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle('Factor vs Diversified Portfolio Dashboard', fontsize=16, fontweight='bold')

    # Panel 1 — Cumulative Returns
    axes[0].plot(factor_cumulative, label='Factor Model', color='blue')
    axes[0].plot(div_cumulative, label='Diversified Model', color='green')
    axes[0].plot(spy_cumulative, label='SPY', color='orange')
    axes[0].set_title('Cumulative Returns')
    axes[0].set_ylabel('Growth of $1')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel 2 — Drawdown Comparison
    factor_dd = factor_cumulative / factor_cumulative.cummax() - 1
    div_dd = div_cumulative / div_cumulative.cummax() - 1
    axes[1].fill_between(factor_dd.index, factor_dd, 0,
                          color='blue', alpha=0.3, label='Factor')
    axes[1].fill_between(div_dd.index, div_dd, 0,
                          color='green', alpha=0.3, label='Diversified')
    axes[1].set_title('Drawdown Comparison')
    axes[1].set_ylabel('Drawdown %')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Panel 3 — Rolling Sharpe
    rolling_factor = factor_portfolio.rolling(63).mean() / \
                     factor_portfolio.rolling(63).std() * np.sqrt(252)
    rolling_div = div_portfolio.rolling(63).mean() / \
                  div_portfolio.rolling(63).std() * np.sqrt(252)
    axes[2].plot(rolling_factor, color='blue', label='Factor')
    axes[2].plot(rolling_div, color='green', label='Diversified')
    axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[2].set_title('Rolling Sharpe (63 days)')
    axes[2].set_ylabel('Sharpe')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()