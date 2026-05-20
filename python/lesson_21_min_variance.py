"""
Multi-Factor Quantitative Backtest — Minimum Variance Portfolio
===============================================================
Lesson 21: Uses eigenvectors of covariance matrix to find minimum
variance weights. Compares equal weight vs minimum variance portfolio.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
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
    print("\nPCA Results:")
    for i, (var, cum) in enumerate(zip(explained, np.cumsum(explained))):
        print(f"PC{i+1}: {var:.2%} variance explained (cumulative: {cum:.2%})")
    loadings = pd.DataFrame(
        pca.components_.T,
        index=returns.columns,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    return pca, explained, loadings

# ── Minimum Variance Weights ───────────────────────────────────────────────────

def min_variance_weights(returns_history, selected_stocks):
    """
    Compute minimum variance weights for selected stocks
    using the covariance matrix eigenvector approach.

    The eigenvector corresponding to the smallest eigenvalue
    of the covariance matrix gives the minimum variance portfolio.

    Steps:
    1. Build covariance matrix of selected stocks
    2. Decompose into eigenvalues + eigenvectors
    3. Take eigenvector with smallest eigenvalue
    4. Normalize weights to sum to 1
    """
    # get historical returns for selected stocks only
    stock_returns = returns_history[selected_stocks].dropna()

    if len(stock_returns) < 10:
        # not enough history — fall back to equal weight
        n = len(selected_stocks)
        return np.array([1/n] * n)

    # build covariance matrix
    cov_matrix = stock_returns.cov().values

    # eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # minimum variance portfolio = eigenvector with smallest eigenvalue
    min_idx = np.argmin(eigenvalues)
    min_var_weights = eigenvectors[:, min_idx]

    # take absolute values (eigenvectors can have negative components)
    # then normalize so weights sum to 1
    min_var_weights = np.abs(min_var_weights)
    min_var_weights = min_var_weights / min_var_weights.sum()

    return min_var_weights

# ── Portfolio Simulation — Equal Weight ───────────────────────────────────────

def run_portfolio_equal(comp, returns, cost_per_trade=0, num_trades=0):
    """Run quarterly rebalance with equal weight."""
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

# ── Portfolio Simulation — Minimum Variance Weight ────────────────────────────

def run_portfolio_minvar(comp, returns, cost_per_trade=0, num_trades=0):
    """
    Run quarterly rebalance with minimum variance weights.
    Uses covariance matrix eigenvector to weight stocks by risk contribution.
    Lower volatility stocks get higher weight.
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

        # compute min variance weights using only data available before rebalance
        # (no look-ahead bias — only use past returns to compute weights)
        weights = min_variance_weights(returns.loc[:date], stocks)
        print(f"  {date.date()} — {stocks} → weights: {weights.round(2)}")

        # apply weights using dot product instead of equal mean
        period_return = returns.loc[start:end, stocks].dot(weights)
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

    # Walk forward testing — equal weight vs minimum variance
    train_start = pd.Timestamp(START_DATE)
    all_equal_returns = []
    all_minvar_returns = []

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

        print(f"\nPeriod {train_end.date()} to {test_end.date()}")

        # ── Equal weight portfolio ─────────────────────────────────────────────
        equal_portfolio = run_portfolio_equal(comp_test, returns_test,
                                              COST_PER_TRADE, NUM_TRADES)
        if equal_portfolio is not None:
            all_equal_returns.append(equal_portfolio)

        # ── Minimum variance portfolio ─────────────────────────────────────────
        minvar_portfolio = run_portfolio_minvar(comp_test, returns_test,
                                                COST_PER_TRADE, NUM_TRADES)
        if minvar_portfolio is not None:
            all_minvar_returns.append(minvar_portfolio)

        train_start += pd.DateOffset(months=TEST_MONTHS)

    # ── Equal weight metrics ───────────────────────────────────────────────────
    equal_portfolio = pd.concat(all_equal_returns)
    equal_cumulative, cagr, sharpe, sortino, calmar, drawdown, var_95 = \
        compute_metrics(equal_portfolio)

    print(f"\nEqual Weight Results:")
    print(f"CAGR:         {cagr:.2%}")
    print(f"Sharpe:       {sharpe:.2f}")
    print(f"Sortino:      {sortino:.2f}")
    print(f"Calmar:       {calmar:.2f}")
    print(f"Max Drawdown: {drawdown:.2%}")
    print(f"VaR (95%):    {var_95:.2%}")

    # ── Minimum variance metrics ───────────────────────────────────────────────
    minvar_portfolio = pd.concat(all_minvar_returns)
    minvar_cumulative, cagr, sharpe, sortino, calmar, drawdown, var_95 = \
        compute_metrics(minvar_portfolio)

    print(f"\nMinimum Variance Results:")
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
    fig.suptitle('Equal Weight vs Minimum Variance Portfolio', fontsize=16, fontweight='bold')

    # Panel 1 — Cumulative Returns
    axes[0].plot(equal_cumulative, label='Equal Weight', color='blue')
    axes[0].plot(minvar_cumulative, label='Min Variance', color='red')
    axes[0].plot(spy_cumulative, label='SPY', color='orange')
    axes[0].set_title('Cumulative Returns')
    axes[0].set_ylabel('Growth of $1')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel 2 — Drawdown Comparison
    equal_dd = equal_cumulative / equal_cumulative.cummax() - 1
    minvar_dd = minvar_cumulative / minvar_cumulative.cummax() - 1
    axes[1].fill_between(equal_dd.index, equal_dd, 0,
                          color='blue', alpha=0.3, label='Equal Weight')
    axes[1].fill_between(minvar_dd.index, minvar_dd, 0,
                          color='red', alpha=0.3, label='Min Variance')
    axes[1].set_title('Drawdown Comparison')
    axes[1].set_ylabel('Drawdown %')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Panel 3 — Rolling Sharpe
    rolling_equal = equal_portfolio.rolling(63).mean() / \
                    equal_portfolio.rolling(63).std() * np.sqrt(252)
    rolling_minvar = minvar_portfolio.rolling(63).mean() / \
                     minvar_portfolio.rolling(63).std() * np.sqrt(252)
    axes[2].plot(rolling_equal, color='blue', label='Equal Weight')
    axes[2].plot(rolling_minvar, color='red', label='Min Variance')
    axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[2].set_title('Rolling Sharpe (63 days)')
    axes[2].set_ylabel('Sharpe')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()