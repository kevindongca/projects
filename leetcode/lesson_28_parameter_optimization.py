"""
Lesson 28 — Optimizing TOP_N and Lookback Windows
===================================================
Grid search over key parameters using walk-forward testing.
Tests multiple combinations of:
  - TOP_N: how many stocks to hold (3, 5, 7, 10)
  - Lookback windows: short/medium/long term signals

Key concept: parameters must be optimized on training data only,
then validated on test data. Optimizing on test data = overfitting.

This is different from in-sample optimization — we use the walk-forward
structure so each parameter set is evaluated on unseen data.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from itertools import product

TICKERS = "AAPL NVDA MSFT GOOGL META TSLA AMD CRM JPM GS AXP V MA AMZN WMT COST NKE JNJ UNH PFE XOM CVX BA CAT HON SHOP"
START_DATE   = '2020-01-01'
END_DATE     = '2026-01-01'
COST_PER_TRADE = 0.001
TRAIN_YEARS  = 2
TEST_MONTHS  = 6

def download_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)
    close = data['Close']
    returns = close.pct_change()
    return close, returns

def compute_signals_custom(close, returns, short_window, medium_window, long_window):
    """
    Compute signals with custom lookback windows.
    short:  reversal lookback
    medium: low vol lookback
    long:   momentum lookback (short+1 to long)
    """
    momentum = close.shift(short_window) / close.shift(long_window) - 1
    reversal = -(close / close.shift(short_window) - 1)
    low_vol  = -(returns.rolling(medium_window).std())
    high_52w = close / close.rolling(long_window).max()
    return momentum, reversal, low_vol, high_52w

def rank_signals(momentum, reversal, low_vol, high_52w):
    r1 = momentum.rank(axis=1, pct=True)
    r2 = reversal.rank(axis=1, pct=True)
    r3 = low_vol.rank(axis=1, pct=True)
    r4 = high_52w.rank(axis=1, pct=True)
    return r1, r2, r3, r4

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

def run_single_backtest(r1, r2, r3, r4, returns, top_n):
    """Run walk-forward backtest for a given parameter set."""
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

        comp = ((r1_t + r2_t + r3_t + r4_t) / 4).dropna()

        holdings = {}
        for date in comp.resample('QE').first().index:
            if date in comp.index:
                holdings[date] = comp.loc[date].nlargest(top_n).index.tolist()

        if not holdings:
            train_start += pd.DateOffset(months=TEST_MONTHS)
            continue

        dates = list(holdings.keys())
        for i, (date, stocks) in enumerate(holdings.items()):
            start = date
            end = dates[i+1] if i+1 < len(dates) else comp.index[-1]
            weights = min_variance_weights(returns.loc[:date], stocks)
            pr = returns_test.loc[start:end, stocks].dot(weights)
            pr.iloc[0] -= COST_PER_TRADE * top_n
            all_returns.append(pr)

        train_start += pd.DateOffset(months=TEST_MONTHS)

    if not all_returns:
        return None, None, None
    portfolio = pd.concat(all_returns)
    cumulative = (1 + portfolio).cumprod()
    years = (cumulative.index[-1] - cumulative.index[0]).days / 365
    cagr  = (cumulative.iloc[-1]) ** (1/years) - 1
    sharpe = portfolio.mean() / portfolio.std() * np.sqrt(252)
    return portfolio, cagr, sharpe

if __name__ == '__main__':
    close, returns = download_data(TICKERS, START_DATE, END_DATE)

    # parameter grid
    top_n_options   = [3, 5, 7, 10]
    window_options  = [
        (10, 42, 126),   # short lookbacks
        (21, 63, 252),   # standard (baseline)
        (21, 126, 252),  # longer medium
        (42, 63, 252),   # longer short
    ]

    results = []
    print("Running parameter grid search...")
    print(f"{'TOP_N':>6} {'Short':>6} {'Med':>6} {'Long':>6} {'CAGR':>8} {'Sharpe':>8}")
    print("-" * 50)

    for top_n, (short, medium, long) in product(top_n_options, window_options):
        momentum, reversal, low_vol, high_52w = compute_signals_custom(
            close, returns, short, medium, long
        )
        r1, r2, r3, r4 = rank_signals(momentum, reversal, low_vol, high_52w)
        portfolio, cagr, sharpe = run_single_backtest(r1, r2, r3, r4, returns, top_n)

        if cagr is not None:
            results.append({
                'top_n': top_n, 'short': short, 'medium': medium,
                'long': long, 'cagr': cagr, 'sharpe': sharpe
            })
            print(f"{top_n:>6} {short:>6} {medium:>6} {long:>6} {cagr:>8.2%} {sharpe:>8.2f}")

    results_df = pd.DataFrame(results).sort_values('sharpe', ascending=False)
    print(f"\nTop 5 parameter combinations by Sharpe:")
    print(results_df.head(5).to_string(index=False))

    best = results_df.iloc[0]
    print(f"\nBest: TOP_N={int(best.top_n)}, windows={int(best.short)}/{int(best.medium)}/{int(best.long)}")
    print(f"CAGR={best.cagr:.2%}, Sharpe={best.sharpe:.2f}")
