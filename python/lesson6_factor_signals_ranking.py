import yfinance as yf
import pandas as pd
import numpy as np

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
print("Top 3 stocks:")
print(top_stocks) 