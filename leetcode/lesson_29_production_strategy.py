"""
================================================================================
Production Multi-Factor Equity Strategy v2.0
================================================================================
Author:  Kevin Dong | github.com/kevindongca | UTSC CS + Stats QF

Incorporates all improvements from Lessons 24-28:
  ✓ Regime detection (200-day MA bear market filter)
  ✓ 6 factors (momentum, reversal, low vol, 52W high, value, quality)
  ✓ Minimum variance portfolio weights (eigendecomposition)
  ✓ Walk-forward testing (2yr train / 6mo test)
  ✓ Transaction costs (0.1% per trade)
  ✓ Monthly rebalancing option
  ✓ Full risk metric suite
  ✓ PCA risk decomposition
  ✓ Professional dashboard

Usage:
    python lesson_29_production_strategy.py
================================================================================
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — tune these parameters
# ══════════════════════════════════════════════════════════════════════════════

TICKERS = (
    "AAPL NVDA MSFT GOOGL META TSLA AMD CRM "
    "JPM GS AXP V MA "
    "AMZN WMT COST NKE "
    "JNJ UNH PFE "
    "XOM CVX "
    "BA CAT HON SHOP"
)

START_DATE     = '2020-01-01'
END_DATE       = '2026-01-01'
TOP_N          = 3
COST_PER_TRADE = 0.001
NUM_TRADES     = TOP_N
TRAIN_YEARS    = 2
TEST_MONTHS    = 6
REBALANCE_FREQ = 'QE'       # 'QE' = quarterly, 'ME' = monthly
USE_REGIME     = True        # enable 200-day MA bear market filter
USE_MINVAR     = True        # enable minimum variance weighting
MA_WINDOW      = 200         # regime detection MA window

# signal lookback windows (trading days)
SHORT_WINDOW  = 21    # ~1 month
MEDIUM_WINDOW = 63    # ~1 quarter
LONG_WINDOW   = 252   # ~1 year

# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════

def download_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)
    close = data['Close']
    returns = close.pct_change()
    return close, returns

def fetch_fundamentals(tickers_str):
    tickers = tickers_str.split()
    pe_dict, roe_dict = {}, {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            pe_dict[ticker]  = info.get('trailingPE', np.nan)
            roe_dict[ticker] = info.get('returnOnEquity', np.nan)
        except:
            pe_dict[ticker] = roe_dict[ticker] = np.nan
    return pd.Series(pe_dict), pd.Series(roe_dict)

# ══════════════════════════════════════════════════════════════════════════════
# SIGNALS
# ══════════════════════════════════════════════════════════════════════════════

def compute_signals(close, returns):
    momentum = close.shift(SHORT_WINDOW) / close.shift(LONG_WINDOW) - 1
    reversal = -(close / close.shift(SHORT_WINDOW) - 1)
    low_vol  = -(returns.rolling(MEDIUM_WINDOW).std())
    high_52w = close / close.rolling(LONG_WINDOW).max()
    return momentum, reversal, low_vol, high_52w

def rank_signals(momentum, reversal, low_vol, high_52w):
    r1 = momentum.rank(axis=1, pct=True)
    r2 = reversal.rank(axis=1, pct=True)
    r3 = low_vol.rank(axis=1, pct=True)
    r4 = high_52w.rank(axis=1, pct=True)
    return r1, r2, r3, r4

def build_composite(r1, r2, r3, r4, pe_series=None, roe_series=None, close=None):
    """Build 4 or 6 factor composite score."""
    if pe_series is not None and roe_series is not None:
        pe_df = pd.DataFrame(
            np.tile(-pe_series.reindex(r1.columns).values, (len(r1), 1)),
            index=r1.index, columns=r1.columns
        )
        roe_df = pd.DataFrame(
            np.tile(roe_series.reindex(r1.columns).values, (len(r1), 1)),
            index=r1.index, columns=r1.columns
        )
        r5 = pe_df.rank(axis=1, pct=True)
        r6 = roe_df.rank(axis=1, pct=True)
        return ((r1 + r2 + r3 + r4 + r5 + r6) / 6).dropna()
    return ((r1 + r2 + r3 + r4) / 4).dropna()

# ══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def min_variance_weights(returns_history, selected_stocks):
    stock_returns = returns_history[selected_stocks].dropna()
    if len(stock_returns) < 10:
        n = len(selected_stocks)
        return np.array([1/n] * n)
    cov_matrix = stock_returns.cov().values
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    min_idx = np.argmin(eigenvalues)
    weights = np.abs(eigenvectors[:, min_idx])
    return weights / weights.sum()

def is_bull_market(date, spy_close, ma_window=200):
    """Returns True if SPY is above its MA on given date."""
    try:
        spy_ma = spy_close.rolling(ma_window).mean()
        return float(spy_close.loc[date]) > float(spy_ma.loc[date])
    except:
        return True

def run_portfolio(comp, returns, spy_close=None,
                  cost_per_trade=0, num_trades=0):
    holdings = {}
    rebalance_dates = comp.resample(REBALANCE_FREQ).first().index
    for date in rebalance_dates:
        if date in comp.index:
            holdings[date] = comp.loc[date].nlargest(TOP_N).index.tolist()

    if not holdings:
        return None

    portfolio_returns = []
    dates = list(holdings.keys())

    for i, (date, stocks) in enumerate(holdings.items()):
        start = date
        end = dates[i+1] if i+1 < len(dates) else comp.index[-1]

        # regime filter
        if USE_REGIME and spy_close is not None:
            if not is_bull_market(date, spy_close, MA_WINDOW):
                pr = pd.Series(0.0, index=returns.loc[start:end].index)
                portfolio_returns.append(pr)
                continue

        # portfolio weights
        if USE_MINVAR:
            weights = min_variance_weights(returns.loc[:date], stocks)
            pr = returns.loc[start:end, stocks].dot(weights)
        else:
            pr = returns.loc[start:end, stocks].mean(axis=1)

        pr.iloc[0] -= cost_per_trade * num_trades
        portfolio_returns.append(pr)

    return pd.concat(portfolio_returns)

# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(portfolio):
    cumulative = (1 + portfolio).cumprod()
    years    = (cumulative.index[-1] - cumulative.index[0]).days / 365
    cagr     = (cumulative.iloc[-1]) ** (1/years) - 1
    sharpe   = portfolio.mean() / portfolio.std() * np.sqrt(252)
    drawdown = (cumulative / cumulative.cummax() - 1).min()
    calmar   = cagr / abs(drawdown)
    sortino  = portfolio.mean() / portfolio[portfolio < 0].std() * np.sqrt(252)
    var_95   = portfolio.quantile(0.05)
    return cumulative, cagr, sharpe, sortino, calmar, drawdown, var_95

def print_metrics(name, cagr, sharpe, sortino, calmar, drawdown, var_95):
    print(f"\n{'─'*45}")
    print(f"  {name}")
    print(f"{'─'*45}")
    print(f"  CAGR:         {cagr:.2%}")
    print(f"  Sharpe:       {sharpe:.2f}")
    print(f"  Sortino:      {sortino:.2f}")
    print(f"  Calmar:       {calmar:.2f}")
    print(f"  Max Drawdown: {drawdown:.2%}")
    print(f"  VaR (95%):    {var_95:.2%}")

# ══════════════════════════════════════════════════════════════════════════════
# PCA
# ══════════════════════════════════════════════════════════════════════════════

def run_pca(returns, n=5):
    clean = returns.dropna()
    scaled = StandardScaler().fit_transform(clean)
    pca = PCA(n_components=n).fit(scaled)
    explained = pca.explained_variance_ratio_
    loadings = pd.DataFrame(pca.components_.T,
                             index=returns.columns,
                             columns=[f'PC{i+1}' for i in range(n)])
    print(f"\n  PCA: PC1={explained[0]:.1%}, PC2={explained[1]:.1%}, "
          f"PC3={explained[2]:.1%} (top 3 explain {sum(explained[:3]):.1%})")
    return explained, loadings

# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

def plot_dashboard(portfolio, cumulative, spy_cumulative, explained, loadings):
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle('Production Multi-Factor Strategy v2.0\nKevin Dong — UTSC',
                 fontsize=15, fontweight='bold')

    # Panel 1 — Cumulative Returns (wide)
    ax1 = fig.add_subplot(3, 2, (1, 2))
    ax1.plot(cumulative, label='Strategy', color='royalblue', linewidth=2)
    ax1.plot(spy_cumulative, label='SPY', color='darkorange', linewidth=1.5)
    ax1.set_title('Cumulative Returns (Walk-Forward Out-of-Sample)')
    ax1.set_ylabel('Growth of $1')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2 — Drawdown
    ax2 = fig.add_subplot(3, 2, 3)
    dd = cumulative / cumulative.cummax() - 1
    spy_dd = spy_cumulative / spy_cumulative.cummax() - 1
    ax2.fill_between(dd.index, dd, 0, color='crimson', alpha=0.4, label='Strategy')
    ax2.fill_between(spy_dd.index, spy_dd, 0, color='orange', alpha=0.2, label='SPY')
    ax2.set_title('Drawdown')
    ax2.set_ylabel('Drawdown %')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 3 — Rolling Sharpe
    ax3 = fig.add_subplot(3, 2, 4)
    rs = portfolio.rolling(63).mean() / portfolio.rolling(63).std() * np.sqrt(252)
    ax3.plot(rs, color='seagreen')
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.axhline(y=1, color='seagreen', linestyle='--', alpha=0.3)
    ax3.set_title('Rolling Sharpe (63 days)')
    ax3.set_ylabel('Sharpe')
    ax3.grid(True, alpha=0.3)

    # Panel 4 — PCA Explained Variance
    ax4 = fig.add_subplot(3, 2, 5)
    ax4.bar(range(1, len(explained)+1), explained, color='steelblue')
    ax4.plot(range(1, len(explained)+1), np.cumsum(explained), 'ro-')
    ax4.set_title('PCA Variance Explained')
    ax4.set_xlabel('Principal Component')
    ax4.grid(True, alpha=0.3)

    # Panel 5 — PC2 Loadings
    ax5 = fig.add_subplot(3, 2, 6)
    pc2 = loadings['PC2'].sort_values()
    colors = ['tomato' if v < 0 else 'steelblue' for v in pc2.values]
    ax5.barh(pc2.index, pc2.values, color=colors)
    ax5.set_title('PC2 Loadings (Value vs Growth)')
    ax5.set_xlabel('Loading')
    ax5.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("="*55)
    print("  Production Multi-Factor Strategy v2.0")
    print("  Kevin Dong — UTSC CS + Stats QF")
    print("="*55)
    print(f"\n  Config: TOP_N={TOP_N}, Regime={USE_REGIME}, MinVar={USE_MINVAR}")
    print(f"  Rebalance: {REBALANCE_FREQ}, Windows: {SHORT_WINDOW}/{MEDIUM_WINDOW}/{LONG_WINDOW}")

    # data
    print("\nDownloading data...")
    close, returns = download_data(TICKERS, START_DATE, END_DATE)

    # signals
    momentum, reversal, low_vol, high_52w = compute_signals(close, returns)
    r1, r2, r3, r4 = rank_signals(momentum, reversal, low_vol, high_52w)

    # fundamentals
    print("Fetching fundamentals...")
    pe_series, roe_series = fetch_fundamentals(TICKERS)

    # SPY for regime + benchmark
    spy_data = yf.download('SPY', start=START_DATE, end=END_DATE)
    spy_close   = spy_data['Close'].squeeze()
    spy_returns = spy_close.pct_change().dropna()
    spy_cum     = (1 + spy_returns).cumprod()

    # PCA
    explained, loadings = run_pca(returns)

    # walk forward
    print("\nRunning walk-forward backtest...")
    train_start = pd.Timestamp(START_DATE)
    all_returns = []

    while True:
        train_end = train_start + pd.DateOffset(years=TRAIN_YEARS)
        test_end  = train_end  + pd.DateOffset(months=TEST_MONTHS)
        if test_end > pd.Timestamp(END_DATE):
            break

        r1_t = r1.loc[train_end:test_end]
        r2_t = r2.loc[train_end:test_end]
        r3_t = r3.loc[train_end:test_end]
        r4_t = r4.loc[train_end:test_end]
        returns_test = returns.loc[train_end:test_end]

        comp = build_composite(r1_t, r2_t, r3_t, r4_t,
                               pe_series, roe_series, close)

        port = run_portfolio(comp, returns, spy_close,
                             COST_PER_TRADE, NUM_TRADES)
        if port is not None:
            all_returns.append(port.loc[train_end:test_end])

        train_start += pd.DateOffset(months=TEST_MONTHS)

    # metrics
    portfolio = pd.concat(all_returns)
    cumulative, *metrics = compute_metrics(portfolio)
    print_metrics("Production Strategy v2.0", *metrics)

    _, spy_cagr, spy_sharpe, spy_sortino, spy_calmar, spy_dd, spy_var = \
        compute_metrics(spy_returns)
    print_metrics("SPY Benchmark", spy_cagr, spy_sharpe,
                  spy_sortino, spy_calmar, spy_dd, spy_var)

    # dashboard
    plot_dashboard(portfolio, cumulative, spy_cum, explained, loadings)
