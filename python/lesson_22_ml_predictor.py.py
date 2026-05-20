"""
Multi-Factor Quantitative Backtest — ML Price Predictor
========================================================
Lesson 22: Uses regression (Gradient Boosting + Ridge) to predict
actual forward returns instead of just direction (classification).
Compares equal weight factor model vs GB vs Ridge regression.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

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

# ── Regression Feature Construction ───────────────────────────────────────────

def build_regression_features(r1, r2, r3, r4, returns, lookahead=63):
    """
    Build feature matrix for regression.
    y = actual forward return (not binary label like in classification).
    Sampled monthly (every 21 days) for speed.
    """
    forward_returns = returns.rolling(lookahead).sum().shift(-lookahead)

    rows = []
    for date in r1.index[::21]:  # sample monthly not daily
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

    return pd.DataFrame(rows).dropna()

# ── ML Return Prediction ───────────────────────────────────────────────────────

def ml_predict_returns(df_train, df_test, top_n=3):
    """
    Train Gradient Boosting and Ridge regression on training data.
    Predict forward returns for test period.
    Select top N stocks by predicted return.

    Gradient Boosting: ensemble of weak decision trees, each correcting
    the errors of the previous one. Better for non-linear relationships.

    Ridge Regression: linear model with L2 regularization to prevent
    overfitting. Simpler but often competitive on small datasets.
    """
    feature_cols = ['mom', 'rev', 'vol', 'h52']

    X_train = df_train[feature_cols]
    y_train = df_train['fwd_return']
    X_test = df_test[feature_cols]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Gradient Boosting Regressor
    gb_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
    gb_model.fit(X_train_scaled, y_train)

    # Ridge Regression
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train_scaled, y_train)

    # feature importance from gradient boosting
    importance = pd.Series(
        gb_model.feature_importances_,
        index=feature_cols
    ).sort_values(ascending=False)
    print(f"  Feature importance: {importance.round(3).to_dict()}")

    df_test = df_test.copy()
    df_test['pred_gb'] = gb_model.predict(X_test_scaled)
    df_test['pred_ridge'] = ridge_model.predict(X_test_scaled)

    # select top N stocks by predicted return on most recent date
    latest = df_test[df_test['date'] == df_test['date'].max()]
    gb_picks = latest.nlargest(top_n, 'pred_gb')['ticker'].tolist()
    ridge_picks = latest.nlargest(top_n, 'pred_ridge')['ticker'].tolist()

    return gb_picks, ridge_picks

# ── Portfolio Simulation ───────────────────────────────────────────────────────

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

def print_metrics(name, cagr, sharpe, sortino, calmar, drawdown, var_95):
    print(f"\n{name} Results:")
    print(f"CAGR:         {cagr:.2%}")
    print(f"Sharpe:       {sharpe:.2f}")
    print(f"Sortino:      {sortino:.2f}")
    print(f"Calmar:       {calmar:.2f}")
    print(f"Max Drawdown: {drawdown:.2%}")
    print(f"VaR (95%):    {var_95:.2%}")

# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # Download data
    close, returns = download_data(TICKERS, START_DATE, END_DATE)

    # Compute and rank signals
    momentum, reversal, low_vol, high_52w = compute_signals(close, returns)
    r1, r2, r3, r4 = rank_signals(momentum, reversal, low_vol, high_52w)

    # Build regression feature DataFrame
    print("Building regression features...")
    reg_df = build_regression_features(r1, r2, r3, r4, returns)
    print(f"Feature DataFrame shape: {reg_df.shape}")

    # Walk forward testing
    train_start = pd.Timestamp(START_DATE)
    all_equal_returns = []
    all_gb_returns = []
    all_ridge_returns = []

    while True:
        train_end = train_start + pd.DateOffset(years=TRAIN_YEARS)
        test_end = train_end + pd.DateOffset(months=TEST_MONTHS)

        if test_end > pd.Timestamp(END_DATE):
            break

        print(f"\nPeriod {train_end.date()} to {test_end.date()}")

        # Slice signals to test period
        r1_test = r1.loc[train_end:test_end]
        r2_test = r2.loc[train_end:test_end]
        r3_test = r3.loc[train_end:test_end]
        r4_test = r4.loc[train_end:test_end]
        returns_test = returns.loc[train_end:test_end]

        comp_test = build_composite([0.25, 0.25, 0.25, 0.25],
                                     r1_test, r2_test, r3_test, r4_test)

        # ── Equal weight factor model ──────────────────────────────────────────
        equal_portfolio = run_portfolio_equal(comp_test, returns_test,
                                              COST_PER_TRADE, NUM_TRADES)
        if equal_portfolio is not None:
            all_equal_returns.append(equal_portfolio)

        # ── ML regression models ───────────────────────────────────────────────
        df_train_reg = reg_df[reg_df['date'] < train_end]
        df_test_reg = reg_df[(reg_df['date'] >= train_end) &
                              (reg_df['date'] < test_end)]

        if len(df_train_reg) > 50 and len(df_test_reg) > 0:
            gb_picks, ridge_picks = ml_predict_returns(
                df_train_reg, df_test_reg, top_n=TOP_N
            )
            print(f"  GB picks:    {gb_picks}")
            print(f"  Ridge picks: {ridge_picks}")

            for picks, storage in [(gb_picks, all_gb_returns),
                                    (ridge_picks, all_ridge_returns)]:
                if picks:
                    pr = returns_test[picks].mean(axis=1)
                    pr.iloc[0] -= COST_PER_TRADE * NUM_TRADES
                    storage.append(pr)

        train_start += pd.DateOffset(months=TEST_MONTHS)

    # ── Compute metrics ────────────────────────────────────────────────────────
    equal_portfolio = pd.concat(all_equal_returns)
    equal_cumulative, *equal_metrics = compute_metrics(equal_portfolio)
    print_metrics("Equal Weight Factor", *equal_metrics)

    gb_portfolio = pd.concat(all_gb_returns)
    gb_cumulative, *gb_metrics = compute_metrics(gb_portfolio)
    print_metrics("Gradient Boosting", *gb_metrics)

    ridge_portfolio = pd.concat(all_ridge_returns)
    ridge_cumulative, *ridge_metrics = compute_metrics(ridge_portfolio)
    print_metrics("Ridge Regression", *ridge_metrics)

    # ── SPY benchmark ──────────────────────────────────────────────────────────
    spy_data = yf.download('SPY', start=START_DATE, end=END_DATE)
    spy_returns = spy_data['Close'].pct_change().dropna()
    spy_cumulative = (1 + spy_returns).cumprod()

    # ── Dashboard ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle('Factor vs ML Regression Portfolio Dashboard',
                 fontsize=16, fontweight='bold')

    # Panel 1 — Cumulative Returns
    axes[0].plot(equal_cumulative, label='Equal Weight', color='blue')
    axes[0].plot(gb_cumulative, label='Gradient Boosting', color='red')
    axes[0].plot(ridge_cumulative, label='Ridge Regression', color='green')
    axes[0].plot(spy_cumulative, label='SPY', color='orange')
    axes[0].set_title('Cumulative Returns')
    axes[0].set_ylabel('Growth of $1')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel 2 — Drawdown Comparison
    equal_dd = equal_cumulative / equal_cumulative.cummax() - 1
    gb_dd = gb_cumulative / gb_cumulative.cummax() - 1
    ridge_dd = ridge_cumulative / ridge_cumulative.cummax() - 1
    axes[1].fill_between(equal_dd.index, equal_dd, 0,
                          color='blue', alpha=0.3, label='Equal Weight')
    axes[1].fill_between(gb_dd.index, gb_dd, 0,
                          color='red', alpha=0.3, label='GB')
    axes[1].fill_between(ridge_dd.index, ridge_dd, 0,
                          color='green', alpha=0.3, label='Ridge')
    axes[1].set_title('Drawdown Comparison')
    axes[1].set_ylabel('Drawdown %')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Panel 3 — Rolling Sharpe
    rolling_equal = equal_portfolio.rolling(63).mean() / \
                    equal_portfolio.rolling(63).std() * np.sqrt(252)
    rolling_gb = gb_portfolio.rolling(63).mean() / \
                 gb_portfolio.rolling(63).std() * np.sqrt(252)
    rolling_ridge = ridge_portfolio.rolling(63).mean() / \
                    ridge_portfolio.rolling(63).std() * np.sqrt(252)
    axes[2].plot(rolling_equal, color='blue', label='Equal Weight')
    axes[2].plot(rolling_gb, color='red', label='GB')
    axes[2].plot(rolling_ridge, color='green', label='Ridge')
    axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[2].set_title('Rolling Sharpe (63 days)')
    axes[2].set_ylabel('Sharpe')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()