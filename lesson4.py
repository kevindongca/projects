import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

df = yf.download("SPY", start='2020-01-01', end="2024-01-01")
close = df["Close"]
returns = close.pct_change().dropna()
print (returns)

#CAGR
n_years = len(returns)/252
cagr= (1+returns).prod() ** (1/n_years) -1
print("CAGR:", round(cagr.iloc[0]*100, 2), "%")

#Sharpe ratio
sharpe=returns.mean() / returns.std() * np.sqrt(252)
print("Sharpe:", round(sharpe.iloc[0], 2))

#Max Drawdown
cumulative=(1+returns).cumprod()
peak = cumulative.cummax()
drawdown = (cumulative - peak) / peak
max_dd= drawdown.min()
print("Max Drawdown:", round(max_dd.iloc[0] * 100, 2), "%")
