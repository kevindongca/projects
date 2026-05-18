"""
Multi-Factor Quantitative Backtest — ML Edition
================================================
Lesson 18: Adds Random Forest classifier alongside factor model.
Compares factor model vs ML model performance side by side.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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

# ── ML Feature Construction ────────────────────────────────────────────────────

def build_ml_features(r1, r2, r3, r4, returns, lookahead=63):
    """
    Build feature matrix and labels for ML training.
    Each row = one stock on one date.
    Features = 4 ranked factor scores.
    Label = 1 if stock beat median forward return, else 0.
    """
    forward_returns = returns.rolling(lookahead).sum().shift(-lookahead)

    rows = []
    for date in r1.index:
        if date not in forward_returns.index:
            continue
        for ticker in r1.columns:
            try:
                row = {
                    'date': date,
                    'ticker': ticker,
                    'mom': r1.loc[date, ticker],
                    'rev': r2.loc[date, ticker],
                    'vol': r3.loc[date, ticker],
                    'h52': r4.loc[date, ticker],
                    'fwd_return': forward_returns.loc[date, ticker]
                }
                rows.append(row)
            except:
                continue

    df = pd.DataFrame(rows).dropna()
    df['label'] = df.groupby('date')['fwd_return'].transform(
        lambda x: (x > x.median()).astype(int)
    )
    return df

# ── ML Stock Selection ─────────────────────────────────────────────────────────

def ml_select_stocks(df_train, df_test, top_n=3):
    """
    Train Random Forest on training data, predict on test data.
    Returns top N stocks by predicted outperformance probability.
    """
    feature_cols = ['mom', 'rev', 'vol', 'h52']

    X_train = df_train[feature_cols]
    y_train = df_train['label']
    X_test = df_test[feature_cols]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    df_test = df_test.copy()
    df_test['prob'] = model.predict_proba(X_test)[:, 1]

    latest = df_test[df_test['date'] == df_test['date'].max()]
    return latest.nlargest(top_n, 'prob')['ticker'].tolist()

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


def run_pca_analysis(returns, n_components=5):
    """
    Decompose return matrix into principal components.
    Shows hidden risk factors driving the portfolio.
    """
    # clean returns — drop NaN rows
    clean_returns = returns.dropna()
    
    # standardize — each stock gets zero mean, unit variance
    scaler = StandardScaler()
    scaled = scaler.fit_transform(clean_returns)
    
    # fit PCA
    pca = PCA(n_components=n_components)
    pca.fit(scaled)
    
    # explained variance — how much each PC explains
    explained = pca.explained_variance_ratio_
    cumulative_explained = np.cumsum(explained)
    
    print("\nPCA Results:")
    for i, (var, cum) in enumerate(zip(explained, cumulative_explained)):
        print(f"PC{i+1}: {var:.2%} variance explained (cumulative: {cum:.2%})")
    
    # component loadings — which stocks drive each PC
    loadings = pd.DataFrame(
        pca.components_.T,
        index=returns.columns,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    
    return pca, explained, loadings

def plot_pca_dashboard(explained, loadings, returns):
    """Plot PCA analysis dashboard."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('PCA Risk Factor Analysis', fontsize=16, fontweight='bold')
    
    # Panel 1 — Explained Variance
    axes[0,0].bar(range(1, len(explained)+1), explained, color='steelblue')
    axes[0,0].plot(range(1, len(explained)+1), 
                   np.cumsum(explained), 'ro-', label='Cumulative')
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
    
    # Panel 3 — PC2 Loadings (second factor)
    pc2 = loadings['PC2'].sort_values()
    axes[1,0].barh(pc2.index, pc2.values, color='purple')
    axes[1,0].set_title('PC2 Loadings (Second Factor)')
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
# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # Download data
    close, returns = download_data(TICKERS, START_DATE, END_DATE)

    # Compute and rank signals
    momentum, reversal, low_vol, high_52w = compute_signals(close, returns)
    r1, r2, r3, r4 = rank_signals(momentum, reversal, low_vol, high_52w)

    # Build full ML feature DataFrame (all dates)
    print("Building ML features...")
    full_df = build_ml_features(r1, r2, r3, r4, returns)

    # Walk forward testing — two portfolios side by side
    train_start = pd.Timestamp(START_DATE)
    all_factor_returns = []
    all_ml_returns = []

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

        # ── Factor model (equal weights) ──────────────────────────────────────
        comp_test = build_composite([0.25, 0.25, 0.25, 0.25],
                                     r1_test, r2_test, r3_test, r4_test)
        factor_portfolio = run_portfolio(comp_test, returns_test,
                                         COST_PER_TRADE, NUM_TRADES)
        if factor_portfolio is not None:
            all_factor_returns.append(factor_portfolio)

        # ── ML model ──────────────────────────────────────────────────────────
        df_train = full_df[full_df['date'] < train_end]
        df_test_ml = full_df[(full_df['date'] >= train_end) &
                              (full_df['date'] < test_end)]

        if len(df_train) > 0 and len(df_test_ml) > 0:
            ml_stocks = ml_select_stocks(df_train, df_test_ml, top_n=TOP_N)
            print(f"Period {train_end.date()} — ML picks: {ml_stocks}")

            if ml_stocks:
                period_return = returns_test[ml_stocks].mean(axis=1)
                period_return.iloc[0] -= COST_PER_TRADE * NUM_TRADES
                all_ml_returns.append(period_return)

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

    # ── ML portfolio metrics ───────────────────────────────────────────────────
    ml_portfolio = pd.concat(all_ml_returns)
    ml_cumulative, cagr, sharpe, sortino, calmar, drawdown, var_95 = \
        compute_metrics(ml_portfolio)

    print(f"\nML Model Results:")
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

    # ── Dashboard — Factor vs ML vs SPY ───────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle('Factor vs ML Portfolio Dashboard', fontsize=16, fontweight='bold')

    # Panel 1 — Cumulative Returns
    axes[0].plot(factor_cumulative, label='Factor Model', color='blue')
    axes[0].plot(ml_cumulative, label='ML Model', color='purple')
    axes[0].plot(spy_cumulative, label='SPY', color='orange')
    axes[0].set_title('Cumulative Returns')
    axes[0].set_ylabel('Growth of $1')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel 2 — Drawdown Comparison
    factor_dd = factor_cumulative / factor_cumulative.cummax() - 1
    ml_dd = ml_cumulative / ml_cumulative.cummax() - 1
    axes[1].fill_between(factor_dd.index, factor_dd, 0,
                          color='blue', alpha=0.3, label='Factor')
    axes[1].fill_between(ml_dd.index, ml_dd, 0,
                          color='purple', alpha=0.3, label='ML')
    axes[1].set_title('Drawdown Comparison')
    axes[1].set_ylabel('Drawdown %')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Panel 3 — Rolling Sharpe
    rolling_factor = factor_portfolio.rolling(63).mean() / \
                     factor_portfolio.rolling(63).std() * np.sqrt(252)
    rolling_ml = ml_portfolio.rolling(63).mean() / \
                 ml_portfolio.rolling(63).std() * np.sqrt(252)
    axes[2].plot(rolling_factor, color='blue', label='Factor')
    axes[2].plot(rolling_ml, color='purple', label='ML')
    axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[2].set_title('Rolling Sharpe (63 days)')
    axes[2].set_ylabel('Sharpe')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    # PCA Analysis
    pca, explained, loadings = run_pca_analysis(returns)
    plot_pca_dashboard(explained, loadings, returns)
    print("\nTop stocks by PC1 loading (market exposure):")
    print(loadings['PC1'].abs().sort_values(ascending=False).head(10))
    print("\nTop stocks by PC2 loading (second factor):")
    print(loadings['PC2'].abs().sort_values(ascending=False).head(10))