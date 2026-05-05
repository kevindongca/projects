import yfinance as yf
import pandas as pd
import numpy as np

tickers = "AAPL MSFT GOOGL AMZN JPM BAC XOM CVX JNJ PFE"
df = yf.download(tickers, start="2018-01-01", end="2023-01-01")
close = df["Close"]
returns = close.pct_change().dropna()

momentum = close.shift(21)/close.shift(252) - 1

reversal=-(close/close.shift()-1)

low_vol= -(returns.rolling(63).std())

last_day=close.index[-1]
row=pd.DataFrame({
    "momentum" : momentum.loc[last_day],
    "reversal": reversal.loc[last_day],
    "low_vol": low_vol.loc[last_day]
})

sorted_row=row.sort_values("momentum", ascending=False)

print(sorted_row)