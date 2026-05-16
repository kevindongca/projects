import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize
COST_PER_TRADE = 0.001  # 0.1% per trade
NUM_TRADES = 3          # buying 3 new stocks each quarter

tickers = "AAPL NVDA MSFT GOOGL META TSLA AMD CRM JPM GS AXP V MA AMZN WMT COST NKE JNJ UNH PFE XOM CVX BA CAT HON SHOP"
data = yf.download(tickers, start = '2020-01-01', end ='2026-01-01')
close = data['Close']

momentum = close.shift(21) / close.shift(252) - 1
reversal =  -(close / close.shift(21) - 1)
returns = close.pct_change()
low_vol = -(returns.rolling(63).std())
high_52w = close / close.rolling(252).max()

r1 = momentum.rank(axis=1, pct=True)
r2 = reversal.rank(axis=1, pct=True)
r3 = low_vol.rank(axis=1, pct=True)
r4 = high_52w.rank(axis=1, pct=True)

def get_sharpe(weights, r1, r2, r3, r4, returns, holdings_dates, composite):
    w1, w2, w3, w4 = weights
    comp = (w1 * r1 + w2 * r2 + w3 * r3 + w4 * r4)
    comp = comp.dropna()
    
    holdings = {}
    quarterly_dates = comp.resample('QE').first().index
    for date in quarterly_dates:
        if date in comp.index:
            row = comp.loc[date]
            top_stocks = row.nlargest(3).index.tolist()
            holdings[date] = top_stocks
    
    if not holdings:
        return 0
    
    portfolio_returns = []
    dates = list(holdings.keys())
    for i, (date, stocks) in enumerate(holdings.items()):
        start = date
        end = dates[i + 1] if i + 1 < len(dates) else comp.index[-1]
        period = returns.loc[start:end, stocks]
        period_return = period.mean(axis=1)
        portfolio_returns.append(period_return)
    
    portfolio = pd.concat(portfolio_returns)
    sharpe = portfolio.mean() / portfolio.std() * np.sqrt(252)
    return -sharpe  # negative because minimize() minimizes, we want to maximize

# ── Walk Forward Testing ───────────────────────────────────────────────────────
train_start = pd.Timestamp('2020-01-01')
all_test_returns = []

while True:
    train_end = train_start + pd.DateOffset(years=2)
    test_end = train_end + pd.DateOffset(months=6)
    
    if test_end > pd.Timestamp('2026-01-01'):
        break
    
    # Slice signals to training period only
    r1_train = r1.loc[train_start:train_end]
    r2_train = r2.loc[train_start:train_end]
    r3_train = r3.loc[train_start:train_end]
    r4_train = r4.loc[train_start:train_end]
    returns_train = returns.loc[train_start:train_end]
    
    # Optimize weights on training period
    result = minimize(get_sharpe, [0.25, 0.25, 0.25, 0.25],
                      args=(r1_train, r2_train, r3_train, r4_train, returns_train, None, None),
                      method='SLSQP',
                      bounds=[(0,1)]*4,
                      constraints={'type': 'eq', 'fun': lambda w: sum(w) - 1})
    
    w1, w2, w3, w4 = result.x
    print(f"Period {train_end.date()} — weights: {w1:.2f} {w2:.2f} {w3:.2f} {w4:.2f}")
    
    # Apply weights to test period
    r1_test = r1.loc[train_end:test_end]
    r2_test = r2.loc[train_end:test_end]
    r3_test = r3.loc[train_end:test_end]
    r4_test = r4.loc[train_end:test_end]
    returns_test = returns.loc[train_end:test_end]
    
    comp_test = (w1*r1_test + w2*r2_test + w3*r3_test + w4*r4_test).dropna()
    
    # Run portfolio on test period
    holdings = {}
    quarterly_dates = comp_test.resample('QE').first().index
    for date in quarterly_dates:
        if date in comp_test.index:
            row = comp_test.loc[date]
            holdings[date] = row.nlargest(3).index.tolist()
    
    if not holdings:
        train_start += pd.DateOffset(months=6)
        continue
    
    dates = list(holdings.keys())
    for i, (date, stocks) in enumerate(holdings.items()):
        start = date
        end = dates[i+1] if i+1 < len(dates) else comp_test.index[-1]
        period = returns_test.loc[start:end, stocks]
        period_return = period.mean(axis=1)
        period_return.iloc[0] -= COST_PER_TRADE * NUM_TRADES
        all_test_returns.append(period_return)
    
    train_start += pd.DateOffset(months=6)

# ── Final Metrics ──────────────────────────────────────────────────────────────
portfolio = pd.concat(all_test_returns)
cumulative = (1 + portfolio).cumprod()

years = (cumulative.index[-1] - cumulative.index[0]).days / 365
cagr = (cumulative.iloc[-1]) ** (1 / years) - 1
sharpe = portfolio.mean() / portfolio.std() * np.sqrt(252)
drawdown = (cumulative / cumulative.cummax() - 1).min()
calmar = cagr / abs(drawdown)
downside_returns = portfolio[portfolio < 0]
sortino = portfolio.mean() / downside_returns.std() * np.sqrt(252)
var_95 = portfolio.quantile(0.05)

print(f"\nWalk Forward Results:")
print(f"CAGR: {cagr:.2%}")
print(f"Sharpe: {sharpe:.2f}")
print(f"Sortino: {sortino:.2f}")
print(f"Calmar: {calmar:.2f}")
print(f"Max Drawdown: {drawdown:.2%}")
print(f"VaR (95%): {var_95:.2%}")

spy_data = yf.download('SPY', start='2020-01-01', end='2026-01-01')
spy_returns = spy_data['Close'].pct_change().dropna()
spy_cumulative = (1 + spy_returns).cumprod()

plt.figure(figsize=(12, 6))
plt.plot(cumulative, label='Walk Forward Portfolio')
plt.plot(spy_cumulative, label='SPY')
plt.legend()
plt.title('Walk Forward Portfolio vs SPY')
plt.show()