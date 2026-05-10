import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import math

tickers = "SHOP NVDA TSLA AMZN MSFT GOOGL META AXP BA WMT"
data = yf.download(tickers, start = '2020-01-01', end ='2026-01-01')
close = data['Close']

momentum = close.shift(21) / close.shift(252) - 1
reversal =  -(close / close.shift(21) - 1)
returns = close.pct_change()
low_vol = -(returns.rolling(63).std())

r1 = momentum.rank(axis=1, pct=True)
r2 = reversal.rank(axis=1, pct=True)
r3 = low_vol.rank(axis=1, pct=True)
composite = (r1 + r2 + r3) / 3
composite = composite.dropna()

holdings = {}
quarterly_dates = composite.resample('QE').first().index

for date in quarterly_dates:
    if date in composite.index:
        row = composite.loc[date]
        top3 = row.nlargest(3).index.tolist()
        holdings[date] = top3

portfolio_returns = []
dates = list(holdings.keys())
for i, (date, tickers) in enumerate(holdings.items()):
    tickers = holdings[date]
    start = date
    end = dates[i + 1] if i + 1 < len(dates) else composite.index[-1]
    
    period = returns.loc[start:end, tickers]
    period_return = period.mean(axis=1)
    portfolio_returns.append(period_return)

portfolio = pd.concat(portfolio_returns)
cumulative = (1 + portfolio).cumprod()

years = (cumulative.index[-1] - cumulative.index[0]).days / 365
cagr = (cumulative.iloc[-1]) ** (1 / years) - 1
sharpe = portfolio.mean() / portfolio.std() * np.sqrt(252)
drawdown = (cumulative / cumulative.cummax() - 1).min()

print(f"CAGR: {cagr:.2%}")
print(f"Sharpe: {sharpe:.2f}")
print(f"Max Drawdown: {drawdown:.2%}")


spy_data = yf.download('SPY', start='2020-01-01', end='2026-01-01')
spy_returns = spy_data['Close'].pct_change().dropna()
spy_cumulative = (1 + spy_returns).cumprod()

plt.figure(figsize=(12, 6))
plt.plot(cumulative, label='Portfolio')
plt.plot(spy_cumulative, label='SPY')
plt.legend()
plt.title('Portfolio vs SPY')
plt.show()