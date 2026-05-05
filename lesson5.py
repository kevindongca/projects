import yfinance as yf
import pandas as pd
import numpy as np

tickers = "AAPL MSFT GOOGL AMZN JPM BAC XOM CVX JNJ PFE"
df = yf.download(tickers, start="2018-01-01", end="2023-01-01")
close = df["Close"]
print(close)
returns = close.pct_change().dropna()
print(returns)

momentum = close.shift(21)/close.shift(252) - 1
print(momentum)

reversal=-(close/close.shift()-1)
print(reversal)

low_vol= -(returns.rolling(63).std())
print(low_vol)