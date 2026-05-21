"""
Lesson 24 — Regime Detection (200-Day MA Filter)
=================================================
When SPY is below its 200-day moving average (bear market regime),
go to cash instead of holding stocks. This protects against sustained
drawdowns like 2022 where the strategy kept losing money.

Key concept: if the broad market is in a downtrend, individual stock
selection doesn't matter much — everything falls together. Better to
sit out and wait for the trend to recover.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

TICKERS = "AAPL NVDA MSFT GOOGL META TSLA AMD CRM JPM GS AXP V MA AMZN WMT COST NKE JNJ UNH PFE XOM CVX BA CAT HON SHOP"
START_DATE = '2020-01-01'
END_DATE   = '2026-01-01'
TOP_N        = 3
COST_PER_TRADE = 0.001
NUM_TRADES   = 3
TRAIN_YEARS  = 2
TEST_MONTHS  = 6

def download_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)
    close = data['Close']
    returns = close.pct_change()
    return close, returns

def compute_signals(close, returns):
    momentum = close.shift(21) / close.shift(252) - 1
    reversal = -(close / close.shift(21) - 1)
    low_vol  = -(returns.rolling(63).std())
    high_52w = close / close.rolling(252).max()
    return momentum, reversal, low_vol, high_52w

def rank_signals(momentum, reversal, low_vol, high_52w):
    r1 = momentum.rank(axis=1, pct=True)
    r2 = reversal.rank(axis=1, pct=True)
    r3 = low_vol.rank(axis=1, pct=True)
    r4 = high_52w.rank(axis=1, pct=True)
    return r1, r2, r3, r4

def build_composite(weights, r1, r2, r3, r4):
    w1, w2, w3, w4 = weights
    return (w1*r1 + w2*r2 + w3*r3 + w4*r4).dropna()

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
    print(f"\n{name}:")
    print(f"  CAGR: {cagr:.2%}  Sharpe: {sharpe:.2f}  Sortino: {sortino:.2f}")
    print(f"  Calmar: {calmar:.2f}  MaxDD: {drawdown:.2%}  VaR: {var_95:.2%}")

def run_portfolio_regime(comp, returns, spy_close,
                         cost_per_trade=0, num_trades=0):
    """
    Run portfolio with regime filter.
    If SPY < 200-day MA on rebalance date → go to cash (return 0).
    If SPY > 200-day MA → invest normally with min variance weights.
    """
    # compute SPY 200-day MA
    spy_ma200 = spy_close.rolling(200).mean()

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

        # find nearest SPY MA value on rebalance date
        try:
            spy_val = spy_close.loc[date]
            ma_val  = spy_ma200.loc[date]
            in_bull_market = float(spy_val) > float(ma_val)
        except:
            in_bull_market = True  # default to invested if data missing

        if in_bull_market:
            # invest with min variance weights
            weights = min_variance_weights(returns.loc[:date], stocks)
            period_return = returns.loc[start:end, stocks].dot(weights)
            period_return.iloc[0] -= cost_per_trade * num_trades
        else:
            # cash — zero return
            period_return = pd.Series(0.0, index=returns.loc[start:end].index)
            print(f"  {date.date()} — CASH (bear market regime)")

        portfolio_returns.append(period_return)

    return pd.concat(portfolio_returns)

if __name__ == '__main__':
    print("Downloading data...")
    close, returns = download_data(TICKERS, START_DATE, END_DATE)

    momentum, reversal, low_vol, high_52w = compute_signals(close, returns)
    r1, r2, r3, r4 = rank_signals(momentum, reversal, low_vol, high_52w)

    # download SPY for regime filter
    spy_data  = yf.download('SPY', start=START_DATE, end=END_DATE)
    spy_close = spy_data['Close'].squeeze()
    spy_returns    = spy_close.pct_change().dropna()
    spy_cumulative = (1 + spy_returns).cumprod()

    train_start = pd.Timestamp(START_DATE)
    all_regime_returns  = []
    all_normal_returns  = []

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

        comp_test = build_composite([0.25,0.25,0.25,0.25],
                                     r1_test, r2_test, r3_test, r4_test)

        # with regime filter
        regime = run_portfolio_regime(comp_test, returns, spy_close,
                                      COST_PER_TRADE, NUM_TRADES)
        if regime is not None:
            all_regime_returns.append(regime.loc[train_end:test_end])

        # without regime filter (baseline)
        holdings = {}
        quarterly_dates = comp_test.resample('QE').first().index
        for date in quarterly_dates:
            if date in comp_test.index:
                holdings[date] = comp_test.loc[date].nlargest(TOP_N).index.tolist()
        dates = list(holdings.keys())
        period_rets = []
        for i, (date, stocks) in enumerate(holdings.items()):
            start = date
            end = dates[i+1] if i+1 < len(dates) else comp_test.index[-1]
            weights = min_variance_weights(returns.loc[:date], stocks)
            pr = returns_test.loc[start:end, stocks].dot(weights)
            pr.iloc[0] -= COST_PER_TRADE * NUM_TRADES
            period_rets.append(pr)
        if period_rets:
            all_normal_returns.append(pd.concat(period_rets))

        train_start += pd.DateOffset(months=TEST_MONTHS)

    regime_portfolio = pd.concat(all_regime_returns)
    normal_portfolio = pd.concat(all_normal_returns)

    regime_cumulative, *regime_m = compute_metrics(regime_portfolio)
    normal_cumulative, *normal_m = compute_metrics(normal_portfolio)

    print_metrics("Min Variance + Regime Filter", *regime_m)
    print_metrics("Min Variance (no filter)", *normal_m)

    _, spy_cagr, spy_sharpe, spy_sortino, spy_calmar, spy_dd, spy_var = \
        compute_metrics(spy_returns)
    print_metrics("SPY", spy_cagr, spy_sharpe, spy_sortino,
                  spy_calmar, spy_dd, spy_var)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle('Regime Detection — Bear Market Cash Filter', fontsize=14, fontweight='bold')

    axes[0].plot(regime_cumulative, label='With Regime Filter', color='green')
    axes[0].plot(normal_cumulative, label='No Filter', color='blue')
    axes[0].plot(spy_cumulative, label='SPY', color='orange')
    axes[0].set_title('Cumulative Returns')
    axes[0].set_ylabel('Growth of $1')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    regime_dd = regime_cumulative / regime_cumulative.cummax() - 1
    normal_dd = normal_cumulative / normal_cumulative.cummax() - 1
    axes[1].fill_between(regime_dd.index, regime_dd, 0,
                          color='green', alpha=0.3, label='Regime Filter')
    axes[1].fill_between(normal_dd.index, normal_dd, 0,
                          color='blue', alpha=0.3, label='No Filter')
    axes[1].set_title('Drawdown Comparison')
    axes[1].set_ylabel('Drawdown %')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
