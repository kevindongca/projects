import yfinance as yf
import pandas as pd
import numpy as np

spy = yf.download("SPY", start="2018-01-01", end="2023-01-01")["Close"].squeeze()
tickers = "AAPL MSFT GOOGL AMZN JPM BAC XOM CVX JNJ PFE"
df = yf.download(tickers, start="2018-01-01", end="2023-01-01")
close = df["Close"]
# print(close)
returns = close.pct_change().dropna()
#print(returns)

momentum = close.shift(21)/close.shift(252) - 1
#print(momentum)

reversal=-(close/close.shift()-1)
#print(reversal)

low_vol= -(returns.rolling(63).std())
#print(low_vol)

# Percentile rank each signal cross-sectionally
momentum_rank = momentum.rank(axis=1, pct=True)
reversal_rank = reversal.rank(axis=1, pct=True)
low_vol_rank = low_vol.rank(axis=1, pct=True)
composite = (momentum_rank + reversal_rank + low_vol_rank) / 3

last_date = composite.iloc[-1]
top_stocks = last_date.nlargest(3)
#print("Top 3 stocks:")
#print(top_stocks) 

# Get quarterly rebalance dates
rebal_dates = composite.resample('QE').last().index
#print(rebal_dates)

# For each rebalance date, pick top 3 stocks
holdings = {}
for date in rebal_dates:
    if date in composite.index:
        scores = composite.loc[date].dropna()
        if len(scores) >= 3:
            top3 = scores.nlargest(3).index.tolist()
            holdings[date] = top3

# Print holdings per period
#for date, stocks in holdings.items():
#    print(f"{date.date()} → {stocks}")

daily_returns = close.pct_change()
port_returns = []

sorted_dates = sorted(holdings.keys())

for i, start_date in enumerate(sorted_dates):
    end_date = sorted_dates[i+1] if  i+1 < len(sorted_dates) else daily_returns.index[-1]
    held = holdings[start_date]
    period = daily_returns.loc[start_date : end_date, held]
    equal_weighted = period.mean(axis=1)
    port_returns.append(equal_weighted)

portfolio= pd.concat(port_returns)
#print(portfolio)

cumulative = (1+portfolio).cumprod()
#print("Final cumulative return:", round((cumulative.iloc[-1]-1)*100, 4), "%")

spy_returns = spy.pct_change().dropna()
spy_cumulative = (1 + spy_returns).cumprod()

# Align to same period as portfolio
spy_aligned = spy_cumulative.reindex(cumulative.index, method='ffill')

print("Strategy return:", round((cumulative.iloc[-1] - 1) * 100, 2), "%")
print("SPY return:", round((spy_aligned.iloc[-1] - 1) * 100, 2), "%")
sharpe = portfolio.mean() / portfolio.std() * np.sqrt(252)
cummax = cumulative.cummax()
max_dd = ((cumulative - cummax) / cummax).min()
print("Sharpe:", round(sharpe, 2))
print("Max Drawdown:", round(max_dd * 100, 2), "%")