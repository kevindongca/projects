import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

df = yf.download("GRAB CTRA SPY", start="2020-01-01", end="2024-01-01")
close = df["Close"]
dr=close.pct_change()
cu=(1+dr).cumprod()
print(cu)
#print ("best performer:"+max(cu)).   --->This is wrong because it shows the max 1 day return in the date range, not the cumulative
print("best performer: " + cu.iloc[-1].idxmax())
#iloc[-1] looks at the last row, idxmax() finds the ticekr with the highest value on that row


cu.plot(title="Cumulative Returns: GRAB vs CTRA vs SPY") #rememebr .plot
plt.ylabel("Growth of $1")
plt.xlabel("Date")
plt.show() #.show