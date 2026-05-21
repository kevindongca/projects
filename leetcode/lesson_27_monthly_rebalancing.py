"""
Lesson 27 — Monthly Rebalancing
================================
Changes rebalance frequency from quarterly to monthly.
More frequent rebalancing captures signals faster but increases
transaction costs. This lesson finds the optimal tradeoff.

Key tradeoff:
- Quarterly: captures multi-month trends, lower costs (1.2%/yr drag)
- Monthly:   captures shorter signals, higher costs (3.6%/yr drag)
- Weekly:    too much noise + costs destroy any edge

We test quarterly vs monthly at different cost levels to find
the break-even transaction cost where monthly stops being worth it.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

TICKERS = "AAPL NVDA MSFT GOOGL META TSLA AMD CRM JPM GS AXP V MA AMZN WMT COST NKE JNJ UNH PFE XOM CVX BA CAT HON SHOP"
START_DATE = '2020-01-01'
END_DATE   = '2026-01-01'
TOP_N      = 3
TRAIN_YEARS = 2
TEST_MONTHS = 6

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

def run_portfolio_frequency(r1, r2, r3, r4, returns,
                             frequency='QE', cost_per_trade=0.001, top_n=3):
    """
    Run portfolio at given rebalance frequency.
    frequency: 'QE' = quarterly, 'ME' = monthly
    """
    comp = ((r1 + r2 + r3 + r4) / 4).dropna()

    holdings = {}
    rebalance_dates = comp.resample(frequency).first().index
    for date in rebalance_dates:
        if date in comp.index:
            holdings[date] = comp.loc[date].nlargest(top_n).index.tolist()

    if not holdings:
        return None

    portfolio_returns = []
    dates = list(holdings.keys())
    for i, (date, stocks) in enumerate(holdings.items()):
        start = date
        end = dates[i+1] if i+1 < len(dates) else comp.index[-1]
        weights = min_variance_weights(returns.loc[:date], stocks)
        pr = returns.loc[start:end, stocks].dot(weights)
        pr.iloc[0] -= cost_per_trade * top_n  # cost on rebalance day
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

if __name__ == '__main__':
    close, returns = download_data(TICKERS, START_DATE, END_DATE)
    momentum, reversal, low_vol, high_52w = compute_signals(close, returns)
    r1, r2, r3, r4 = rank_signals(momentum, reversal, low_vol, high_52w)

    # test different cost levels for monthly rebalancing
    cost_levels = [0.0005, 0.001, 0.002, 0.005]
    results = {}

    # quarterly baseline at 0.1% cost
    q_port = run_portfolio_frequency(r1, r2, r3, r4, returns,
                                      frequency='QE', cost_per_trade=0.001)
    q_cum, q_cagr, q_sharpe, *_ = compute_metrics(q_port)
    print(f"Quarterly (0.10% cost): CAGR={q_cagr:.2%}, Sharpe={q_sharpe:.2f}")
    results['Quarterly 0.10%'] = q_cum

    for cost in cost_levels:
        m_port = run_portfolio_frequency(r1, r2, r3, r4, returns,
                                          frequency='ME', cost_per_trade=cost)
        m_cum, m_cagr, m_sharpe, *_ = compute_metrics(m_port)
        label = f"Monthly {cost*100:.2f}%"
        print(f"{label}: CAGR={m_cagr:.2%}, Sharpe={m_sharpe:.2f}")
        results[label] = m_cum

    spy_data = yf.download('SPY', start=START_DATE, end=END_DATE)
    spy_rets = spy_data['Close'].squeeze().pct_change().dropna()
    spy_cum  = (1 + spy_rets).cumprod()

    plt.figure(figsize=(14, 7))
    colors = ['blue', 'green', 'limegreen', 'orange', 'red']
    for (label, cum), color in zip(results.items(), colors):
        plt.plot(cum, label=label, color=color)
    plt.plot(spy_cum, label='SPY', color='black', linestyle='--')
    plt.title('Rebalance Frequency vs Transaction Cost Tradeoff')
    plt.ylabel('Growth of $1')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
