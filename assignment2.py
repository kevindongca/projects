import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

df = yf.download("SPY GRAB CTRA TSLA NVDA", start='2020-01-01', end="2024-01-01")
close=df["Close"]
returns=close.pct_change().dropna()

#calcualte CAGR
n_years=len(returns)/252
cagr=(1+returns).prod() ** (1/n_years) - 1

#Sharpe Ratio
sharpe=returns.mean() / returns.std() * np.sqrt(252)

#Max drawdrown
cumulative=(1+returns).cumprod()
peak = cumulative.cummax()
drawdown = (cumulative - peak) / peak
max_dd= drawdown.min()

#print all:
print("CAGR:", cagr)
print("Sharpe:", sharpe)
print("Max Drawdown:", max_dd)

#Best Sharpe Ratio
print("Best Sharpe:", sharpe.idxmax())

#plot returns on a graph
cumulative.plot(title="Comparison of 5 stocks")
plt.ylabel("Growth of $1")
plt.xlabel("Date")
plt.show()