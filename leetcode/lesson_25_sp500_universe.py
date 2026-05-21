"""
Lesson 25 — Expanding to S&P 500 Universe
==========================================
Replaces 26 stocks with a larger universe of S&P 500 constituents.
More stocks = better signal quality and more diversification.
We use a curated list of 100 liquid large-caps as a proxy for the
full S&P 500 (downloading all 500 would take too long).

Key improvement: with 100 stocks, the top 3 picks by composite score
are genuinely differentiated. With 26 stocks many have similar scores.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# 100 liquid large-cap US stocks across all sectors
TICKERS = (
    # Tech
    "AAPL MSFT NVDA GOOGL META TSLA AMD INTC CRM ADBE ORCL QCOM "
    "TXN AMAT LRCX MU KLAC SNPS CDNS PANW "
    # Financials
    "JPM BAC GS MS WFC C BLK AXP V MA COF USB TFC "
    # Consumer
    "AMZN WMT COST TGT HD LOW NKE SBUX MCD YUM "
    # Healthcare
    "JNJ UNH PFE MRK ABBV TMO ABT DHR BMY AMGN "
    # Energy
    "XOM CVX COP SLB EOG PSX MPC "
    # Industrials
    "BA CAT HON GE UPS FDX RTX LMT "
    # Communications
    "T VZ DIS NFLX CMCSA "
    # Materials & Real Estate
    "LIN APD NEM SHOP"
)

START_DATE   = '2020-01-01'
END_DATE     = '2026-01-01'
TOP_N        = 5        # hold 5 stocks with larger universe
COST_PER_TRADE = 0.001
NUM_TRADES   = 5
TRAIN_YEARS  = 2
TEST_MONTHS  = 6

def download_data(tickers, start, end):
    """Download with auto_adjust to handle delistings gracefully."""
    data = yf.download(tickers, start=start, end=end)
    close = data['Close']
    # drop columns with too many NaNs (recently listed or delisted stocks)
    close = close.dropna(axis=1, thresh=int(len(close)*0.8))
    returns = close.pct_change()
    print(f"Universe size after cleaning: {len(close.columns)} stocks")
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
        period_return = returns.loc[start:end, stocks].dot(weights)
        period_return.iloc[0] -= cost_per_trade * num_trades
        portfolio_returns.append(period_return)

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
    print(f"  CAGR: {cagr:.2%}  Sharpe: {sharpe:.2f}  Sortino: {sortino:.2f}")
    print(f"  Calmar: {calmar:.2f}  MaxDD: {drawdown:.2%}  VaR: {var_95:.2%}")

if __name__ == '__main__':
    print("Downloading 100-stock universe...")
    close, returns = download_data(TICKERS, START_DATE, END_DATE)

    momentum, reversal, low_vol, high_52w = compute_signals(close, returns)
    r1, r2, r3, r4 = rank_signals(momentum, reversal, low_vol, high_52w)

    train_start = pd.Timestamp(START_DATE)
    all_returns = []

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

        port = run_portfolio(comp_test, returns_test, COST_PER_TRADE, NUM_TRADES)
        if port is not None:
            all_returns.append(port)

        train_start += pd.DateOffset(months=TEST_MONTHS)

    portfolio = pd.concat(all_returns)
    cumulative, *metrics = compute_metrics(portfolio)
    print_metrics("100-Stock Universe (Top 5, Min Variance)", *metrics)

    spy_data = yf.download('SPY', start=START_DATE, end=END_DATE)
    spy_returns    = spy_data['Close'].squeeze().pct_change().dropna()
    spy_cumulative = (1 + spy_returns).cumprod()

    plt.figure(figsize=(14, 6))
    plt.plot(cumulative, label='100-Stock Strategy', color='blue')
    plt.plot(spy_cumulative, label='SPY', color='orange')
    plt.title('100-Stock Universe vs SPY')
    plt.ylabel('Growth of $1')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
