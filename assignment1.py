import pandas as pd

data={
    "ticker": ["APPL", "GOOGL", "AMZN", "MSFT", "TSLA"],
    "price": [175.0, 380.0, 140.0, 300.0, 800.0],
    "shares": [10, 5, 8, 12, 3]
}
df=pd.DataFrame(data)
df["market values"] = df["price"] * df["shares"]
df["weight"] = df["market values"] / df["market values"].sum()
print(df.sort_values("weight", ascending=False))
df_sorted = df.sort_values("weight", ascending=False).reset_index(drop=True)
print(df_sorted)
print(df_sorted.iloc[0]["ticker"])