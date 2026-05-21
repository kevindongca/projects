"""
Lesson 26 — Fundamental Factors (P/E and ROE via yfinance)
===========================================================
Adds value (P/E ratio) and quality (ROE) as factors 5 and 6.
These are fundamental signals uncorrelated with price signals,
so they add genuine diversification of alpha.

Value signal:  low P/E = undervalued = buy (negative of P/E)
Quality signal: high ROE = profitable company = buy

Note: yfinance fundamentals are point-in-time snapshots, not
historical time series. This introduces some look-ahead bias.
For production use, Bloomberg or Polygon.io provide historical
point-in-time fundamental data.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

TICKERS = "AAPL NVDA MSFT GOOGL META TSLA AMD CRM JPM GS AXP V MA AMZN WMT COST NKE JNJ UNH PFE XOM CVX BA CAT HON SHOP"
START_DATE   = '2020-01-01'
END_DATE     = '2026-01-01'
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

def fetch_fundamentals(tickers_str):
    """
    Fetch P/E ratio and ROE for each ticker using yfinance.
    Returns two Series indexed by ticker.
    Note: these are current values, not historical time series.
    """
    tickers = tickers_str.split()
    pe_dict  = {}
    roe_dict = {}

    print("Fetching fundamental data...")
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            pe  = info.get('trailingPE', np.nan)
            roe = info.get('returnOnEquity', np.nan)
            pe_dict[ticker]  = pe
            roe_dict[ticker] = roe
            print(f"  {ticker}: P/E={pe:.1f}, ROE={roe:.2%}" if pe and roe else f"  {ticker}: missing data")
        except:
            pe_dict[ticker]  = np.nan
            roe_dict[ticker] = np.nan

    return pd.Series(pe_dict), pd.Series(roe_dict)

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

def build_composite_6factor(r1, r2, r3, r4, pe_series, roe_series, close):
    """
    Build 6-factor composite score.
    Factors 1-4: price signals (ranked cross-sectionally daily)
    Factor 5:    value — negative P/E ranked (lower P/E = higher rank)
    Factor 6:    quality — ROE ranked (higher ROE = higher rank)

    Since P/E and ROE are static snapshots, we broadcast them across
    all dates as constant columns.
    """
    # align fundamentals to close columns
    pe_aligned  = pe_series.reindex(close.columns)
    roe_aligned = roe_series.reindex(close.columns)

    # create DataFrame with same index as r1 (broadcast static values)
    pe_df  = pd.DataFrame(
        np.tile(-pe_aligned.values, (len(r1), 1)),  # negative P/E = value signal
        index=r1.index, columns=r1.columns
    )
    roe_df = pd.DataFrame(
        np.tile(roe_aligned.values, (len(r1), 1)),
        index=r1.index, columns=r1.columns
    )

    # rank fundamentals cross-sectionally
    r5 = pe_df.rank(axis=1, pct=True)   # value
    r6 = roe_df.rank(axis=1, pct=True)  # quality

    # equal weight all 6 factors
    composite = (r1 + r2 + r3 + r4 + r5 + r6) / 6
    return composite.dropna()

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

def run_portfolio(comp, returns, cost_per_trade=0, num_trades=0):
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
        weights = min_variance_weights(returns.loc[:date], stocks)
        pr = returns.loc[start:end, stocks].dot(weights)
        pr.iloc[0] -= cost_per_trade * num_trades
        portfolio_returns.append(pr)
    return pd.concat(portfolio_returns)

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
    print(f"  CAGR: {cagr:.2%}  Sharpe: {sharpe:.2f}  MaxDD: {drawdown:.2%}")

if __name__ == '__main__':
    close, returns = download_data(TICKERS, START_DATE, END_DATE)
    momentum, reversal, low_vol, high_52w = compute_signals(close, returns)
    r1, r2, r3, r4 = rank_signals(momentum, reversal, low_vol, high_52w)

    # fetch fundamentals
    pe_series, roe_series = fetch_fundamentals(TICKERS)
    print(f"\nP/E range: {pe_series.min():.1f} to {pe_series.max():.1f}")
    print(f"ROE range: {roe_series.min():.2%} to {roe_series.max():.2%}")

    train_start = pd.Timestamp(START_DATE)
    all_6f_returns = []
    all_4f_returns = []

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

        # 6 factor composite
        comp_6f = build_composite_6factor(
            r1_t, r2_t, r3_t, r4_t, pe_series, roe_series, close
        )
        port_6f = run_portfolio(comp_6f, returns, COST_PER_TRADE, NUM_TRADES)
        if port_6f is not None:
            all_6f_returns.append(port_6f.loc[train_end:test_end])

        # 4 factor composite (baseline)
        comp_4f = ((r1_t + r2_t + r3_t + r4_t) / 4).dropna()
        port_4f = run_portfolio(comp_4f, returns_test, COST_PER_TRADE, NUM_TRADES)
        if port_4f is not None:
            all_4f_returns.append(port_4f)

        train_start += pd.DateOffset(months=TEST_MONTHS)

    p6 = pd.concat(all_6f_returns)
    p4 = pd.concat(all_4f_returns)

    c6, *m6 = compute_metrics(p6)
    c4, *m4 = compute_metrics(p4)

    print_metrics("6-Factor (+ Value + Quality)", *m6)
    print_metrics("4-Factor (baseline)", *m4)

    spy_data = yf.download('SPY', start=START_DATE, end=END_DATE)
    spy_rets = spy_data['Close'].squeeze().pct_change().dropna()
    spy_cum  = (1 + spy_rets).cumprod()

    plt.figure(figsize=(14, 6))
    plt.plot(c6, label='6-Factor', color='green')
    plt.plot(c4, label='4-Factor', color='blue')
    plt.plot(spy_cum, label='SPY', color='orange')
    plt.title('6-Factor (Value + Quality) vs 4-Factor vs SPY')
    plt.ylabel('Growth of $1')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
