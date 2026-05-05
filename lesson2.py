import yfinance as yf
import pandas as pd

df = yf.download("AAPL", start="2020-01-01", end="2024-01-01")
print(df)

close = df["Close"]
print(close)

returns=close.pct_change()
print(returns)

cumulative = (1+returns).cumprod()
print(cumulative)