"""
================================================================================
Multi-Factor Quantitative Equity Strategy
================================================================================
Author:      Kevin Dong
GitHub:      github.com/kevindongca
Program:     CS Specialist (AI) + Statistics Major (QF) — UTSC

Strategy Overview:
    A systematic, factor-based equity strategy that selects and quarterly
    rebalances a concentrated portfolio of 3 stocks from a 26-stock universe
    using 4 price-based signals: momentum, reversal, low volatility, and
    52-week high proximity. Portfolio weights are determined by minimum variance
    optimization using covariance matrix eigendecomposition. Strategy is
    validated using walk-forward testing with transaction costs.

Universe:    26 large-cap US equities across 6 sectors
Factors:     Momentum (12M-1M), Reversal (1M), Low Volatility (3M), 52W High
Weighting:   Minimum variance (eigenvector of covariance matrix)
Rebalance:   Quarterly
Backtest:    Walk-forward (2yr train / 6mo test windows)
Costs:       0.1% per trade (3 trades per quarter = 0.3% quarterly drag)

Walk-Forward Results (2022-2026, out-of-sample):
    CAGR:         8.18%
    Sharpe:       0.94
    Sortino:      1.33
    Calmar:       0.39
    Max Drawdown: -21.06%
    VaR (95%):    -2.01%
================================================================================
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

TICKERS = (
    "AAPL NVDA MSFT GOOGL META TSLA AMD CRM "   # Technology
    "JPM GS AXP V MA "                           # Financials
    "AMZN WMT COST NKE "                         # Consumer
    "JNJ UNH PFE "                               # Healthcare
    "XOM CVX "                                   # Energy
    "BA CAT HON SHOP"                            # Industrials + other
)

START_DATE   = '2020-01-01'
END_DATE     = '2026-01-01'
TOP_N        = 3        # stocks to hold each quarter
COST_PER_TRADE = 0.001  # 0.1% transaction cost per trade
NUM_TRADES   = 3        # stocks bought each rebalance
TRAIN_YEARS  = 2        # walk-forward training window
TEST_MONTHS  = 6        # walk-forward test window

# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════

def download_data(tickers, start, end):
    """Download historical closing prices and compute daily returns."""
    data = yf.download(tickers, start=start, end=end)
    close = data['Close']
    returns = close.pct_change()
    return close, returns

# ══════════════════════════════════════════════════════════════════════════════
# FACTOR SIGNALS
# ══════════════════════════════════════════════════════════════════════════════

def compute_signals(close, returns):
    """
    Compute 4 price-based factor signals.

    Momentum  (Jegadeesh & Titman 1993):
        12-month return excluding last month. Stocks that trended up over
        the past year tend to continue outperforming.

    Reversal  (De Bondt & Thaler 1985):
        Negative of last month return. Short-term losers tend to bounce back
        due to investor overreaction.

    Low Vol   (Blitz & Van Vliet 2007):
        Negative of 3-month rolling volatility. Less volatile stocks
        outperform on a risk-adjusted basis due to institutional constraints.

    52W High  (George & Hwang 2004):
        Price relative to 52-week high. Stocks near their 52-week high tend
        to break out and continue higher.
    """
    momentum = close.shift(21) / close.shift(252) - 1
    reversal = -(close / close.shift(21) - 1)
    low_vol  = -(returns.rolling(63).std())
    high_52w = close / close.rolling(252).max()
    return momentum, reversal, low_vol, high_52w

def rank_signals(momentum, reversal, low_vol, high_52w):
    """
    Percentile rank each signal cross-sectionally (across tickers per day).
    Ranking normalizes signals to [0,1] so they contribute equally to composite.
    """
    r1 = momentum.rank(axis=1, pct=True)
    r2 = reversal.rank(axis=1, pct=True)
    r3 = low_vol.rank(axis=1, pct=True)
    r4 = high_52w.rank(axis=1, pct=True)
    return r1, r2, r3, r4

def build_composite(weights, r1, r2, r3, r4):
    """Weighted average of ranked signals into a single composite score."""
    w1, w2, w3, w4 = weights
    return (w1*r1 + w2*r2 + w3*r3 + w4*r4).dropna()

# ══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def min_variance_weights(returns_history, selected_stocks):
    """
    Compute minimum variance portfolio weights using eigendecomposition.

    The eigenvector corresponding to the smallest eigenvalue of the
    covariance matrix defines the portfolio with minimum variance.
    Stocks with higher historical volatility receive lower weights.
    """
    stock_returns = returns_history[selected_stocks].dropna()

    if len(stock_returns) < 10:
        n = len(selected_stocks)
        return np.array([1/n] * n)

    cov_matrix = stock_returns.cov().values
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    min_idx = np.argmin(eigenvalues)
    weights = np.abs(eigenvectors[:, min_idx])
    return weights / weights.sum()

def run_portfolio(comp, returns, cost_per_trade=0, num_trades=0,
                  use_min_variance=True):
    """
    Quarterly rebalance portfolio simulation.

    Stock selection: top N by composite factor score.
    Weighting:       minimum variance (default) or equal weight.
    Costs:           applied on first day of each holding period.
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
        end = dates[i+1] if i+1 < len(dates) else comp.index[-1]

        if use_min_variance:
            weights = min_variance_weights(returns.loc[:date], stocks)
            period_return = returns.loc[start:end, stocks].dot(weights)
        else:
            period_return = returns.loc[start:end, stocks].mean(axis=1)

        period_return.iloc[0] -= cost_per_trade * num_trades
        portfolio_returns.append(period_return)

    return pd.concat(portfolio_returns)

# ══════════════════════════════════════════════════════════════════════════════
# RISK METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(portfolio):
    """
    Compute full risk metric suite from daily return series.

    CAGR:         Compound Annual Growth Rate
    Sharpe:       Annualized return / annualized volatility
    Sortino:      Return / downside volatility only
    Calmar:       CAGR / absolute max drawdown
    Max Drawdown: Worst peak-to-trough decline
    VaR (95%):    5th percentile of daily returns
    """
    cumulative = (1 + portfolio).cumprod()
    years      = (cumulative.index[-1] - cumulative.index[0]).days / 365
    cagr       = (cumulative.iloc[-1]) ** (1/years) - 1
    sharpe     = portfolio.mean() / portfolio.std() * np.sqrt(252)
    drawdown   = (cumulative / cumulative.cummax() - 1).min()
    calmar     = cagr / abs(drawdown)
    sortino    = portfolio.mean() / portfolio[portfolio < 0].std() * np.sqrt(252)
    var_95     = portfolio.quantile(0.05)
    return cumulative, cagr, sharpe, sortino, calmar, drawdown, var_95

def print_metrics(name, cagr, sharpe, sortino, calmar, drawdown, var_95):
    """Pretty print risk metrics."""
    print(f"\n{'─'*40}")
    print(f"  {name}")
    print(f"{'─'*40}")
    print(f"  CAGR:         {cagr:.2%}")
    print(f"  Sharpe:       {sharpe:.2f}")
    print(f"  Sortino:      {sortino:.2f}")
    print(f"  Calmar:       {calmar:.2f}")
    print(f"  Max Drawdown: {drawdown:.2%}")
    print(f"  VaR (95%):    {var_95:.2%}")

# ══════════════════════════════════════════════════════════════════════════════
# PCA ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def run_pca_analysis(returns, n_components=5):
    """
    Decompose return covariance matrix into principal components.
    Reveals hidden risk factors driving the portfolio universe.
    """
    clean_returns = returns.dropna()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(clean_returns)
    pca = PCA(n_components=n_components)
    pca.fit(scaled)
    explained = pca.explained_variance_ratio_

    print(f"\n{'─'*40}")
    print("  PCA Risk Factor Decomposition")
    print(f"{'─'*40}")
    for i, (var, cum) in enumerate(zip(explained, np.cumsum(explained))):
        print(f"  PC{i+1}: {var:.2%} explained  (cumulative: {cum:.2%})")

    loadings = pd.DataFrame(
        pca.components_.T,
        index=returns.columns,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    return pca, explained, loadings

# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

def plot_main_dashboard(portfolio, cumulative, spy_cumulative, spy_returns):
    """3-panel performance dashboard: returns, drawdown, rolling Sharpe."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle('Multi-Factor Minimum Variance Strategy',
                 fontsize=16, fontweight='bold')

    # Panel 1 — Cumulative Returns
    axes[0].plot(cumulative, label='Strategy', color='royalblue', linewidth=1.5)
    axes[0].plot(spy_cumulative, label='SPY', color='darkorange', linewidth=1.5)
    axes[0].set_title('Cumulative Returns (Walk-Forward Out-of-Sample)')
    axes[0].set_ylabel('Growth of $1')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel 2 — Drawdown
    drawdown_series = cumulative / cumulative.cummax() - 1
    axes[1].fill_between(drawdown_series.index, drawdown_series, 0,
                          color='crimson', alpha=0.4, label='Strategy DD')
    spy_dd = spy_cumulative / spy_cumulative.cummax() - 1
    axes[1].fill_between(spy_dd.index, spy_dd, 0,
                          color='darkorange', alpha=0.2, label='SPY DD')
    axes[1].set_title('Drawdown')
    axes[1].set_ylabel('Drawdown %')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Panel 3 — Rolling Sharpe
    rolling_sharpe = (portfolio.rolling(63).mean() /
                      portfolio.rolling(63).std() * np.sqrt(252))
    axes[2].plot(rolling_sharpe, color='seagreen', linewidth=1.2)
    axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[2].axhline(y=1, color='seagreen', linestyle='--', alpha=0.3,
                    label='Sharpe = 1')
    axes[2].set_title('Rolling Sharpe Ratio (63-day window)')
    axes[2].set_ylabel('Sharpe')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_pca_dashboard(explained, loadings, returns):
    """4-panel PCA risk factor analysis dashboard."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('PCA Risk Factor Analysis', fontsize=16, fontweight='bold')

    # Panel 1 — Explained Variance
    axes[0,0].bar(range(1, len(explained)+1), explained, color='steelblue')
    axes[0,0].plot(range(1, len(explained)+1), np.cumsum(explained),
                   'ro-', label='Cumulative')
    axes[0,0].set_title('Variance Explained by Each PC')
    axes[0,0].set_xlabel('Principal Component')
    axes[0,0].set_ylabel('Variance Explained')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

    # Panel 2 — PC1 Loadings (market factor)
    pc1 = loadings['PC1'].sort_values()
    axes[0,1].barh(pc1.index, pc1.values, color='steelblue')
    axes[0,1].set_title('PC1 Loadings (Market Factor)')
    axes[0,1].set_xlabel('Loading')
    axes[0,1].grid(True, alpha=0.3)

    # Panel 3 — PC2 Loadings (value vs growth)
    pc2 = loadings['PC2'].sort_values()
    axes[1,0].barh(pc2.index, pc2.values, color='purple')
    axes[1,0].set_title('PC2 Loadings (Value vs Growth)')
    axes[1,0].set_xlabel('Loading')
    axes[1,0].grid(True, alpha=0.3)

    # Panel 4 — Correlation heatmap
    corr = returns.dropna().corr()
    im = axes[1,1].imshow(corr.values, cmap='RdYlGn', vmin=-1, vmax=1)
    axes[1,1].set_xticks(range(len(corr.columns)))
    axes[1,1].set_yticks(range(len(corr.columns)))
    axes[1,1].set_xticklabels(corr.columns, rotation=90, fontsize=7)
    axes[1,1].set_yticklabels(corr.columns, fontsize=7)
    axes[1,1].set_title('Return Correlation Matrix')
    plt.colorbar(im, ax=axes[1,1])

    plt.tight_layout()
    plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    print("="*60)
    print("  Multi-Factor Minimum Variance Strategy")
    print("  Kevin Dong — UTSC CS + Stats QF")
    print("="*60)

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\nDownloading data...")
    close, returns = download_data(TICKERS, START_DATE, END_DATE)

    # ── Signals ───────────────────────────────────────────────────────────────
    print("Computing factor signals...")
    momentum, reversal, low_vol, high_52w = compute_signals(close, returns)
    r1, r2, r3, r4 = rank_signals(momentum, reversal, low_vol, high_52w)

    # ── PCA ───────────────────────────────────────────────────────────────────
    pca, explained, loadings = run_pca_analysis(returns)

    # ── Walk Forward Testing ──────────────────────────────────────────────────
    print("\nRunning walk-forward backtest...")
    train_start = pd.Timestamp(START_DATE)
    all_minvar_returns  = []
    all_equal_returns   = []

    while True:
        train_end = train_start + pd.DateOffset(years=TRAIN_YEARS)
        test_end  = train_end  + pd.DateOffset(months=TEST_MONTHS)

        if test_end > pd.Timestamp(END_DATE):
            break

        r1_test = r1.loc[train_end:test_end]
        r2_test = r2.loc[train_end:test_end]
        r3_test = r3.loc[train_end:test_end]
        r4_test = r4.loc[train_end:test_end]
        returns_test = returns.loc[train_end:test_end]

        comp_test = build_composite([0.25, 0.25, 0.25, 0.25],
                                     r1_test, r2_test, r3_test, r4_test)

        # Minimum variance portfolio
        minvar = run_portfolio(comp_test, returns_test,
                               COST_PER_TRADE, NUM_TRADES,
                               use_min_variance=True)
        if minvar is not None:
            all_minvar_returns.append(minvar)

        # Equal weight portfolio (benchmark comparison)
        equal = run_portfolio(comp_test, returns_test,
                              COST_PER_TRADE, NUM_TRADES,
                              use_min_variance=False)
        if equal is not None:
            all_equal_returns.append(equal)

        train_start += pd.DateOffset(months=TEST_MONTHS)

    # ── Metrics ───────────────────────────────────────────────────────────────
    minvar_portfolio = pd.concat(all_minvar_returns)
    equal_portfolio  = pd.concat(all_equal_returns)

    minvar_cumulative, *minvar_m = compute_metrics(minvar_portfolio)
    equal_cumulative,  *equal_m  = compute_metrics(equal_portfolio)

    print_metrics("Min Variance Strategy (Primary)", *minvar_m)
    print_metrics("Equal Weight Strategy (Baseline)", *equal_m)

    # ── SPY Benchmark ─────────────────────────────────────────────────────────
    spy_data = yf.download('SPY', start=START_DATE, end=END_DATE)
    spy_returns    = spy_data['Close'].pct_change().dropna().squeeze()
    spy_cumulative = (1 + spy_returns).cumprod()
    spy_cumulative_check, spy_cagr, spy_sharpe, spy_sortino, spy_calmar, spy_dd, spy_var = \
        compute_metrics(spy_returns.squeeze())
    print_metrics("SPY Benchmark", spy_cagr, spy_sharpe,
                  spy_sortino, spy_calmar, spy_dd, spy_var)

    # ── PCA Insights ──────────────────────────────────────────────────────────
    print(f"\n{'─'*40}")
    print("  Top stocks by PC1 (market exposure):")
    print(loadings['PC1'].abs().sort_values(ascending=False).head(5).to_string())
    print(f"\n  PC2 split — Value/Energy (positive):")
    print(loadings['PC2'].sort_values(ascending=False).head(5).to_string())
    print(f"\n  PC2 split — Tech/Growth (negative):")
    print(loadings['PC2'].sort_values(ascending=True).head(5).to_string())

    # ── Dashboards ────────────────────────────────────────────────────────────
    plot_main_dashboard(minvar_portfolio, minvar_cumulative,
                        spy_cumulative, spy_returns)
    plot_pca_dashboard(explained, loadings, returns)