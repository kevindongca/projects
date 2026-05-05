import pandas as pd

data={
    "stock": ["APPL", "GOOGL", "AMZN"],
    "price": [175.0, 380.0, 140.0],
    "shares": [10, 5, 8]
}
df=pd.DataFrame(data)
print(df)

print(df["price"]) #this will print the price column

df["market value"] = df["price"] * df["shares"] #this will create a new column called market value by multiplying price and shares
print(df)

print(df[df["market value"]>1000]) #this will print the rows where the market value is greater than 1000

print(df.sort_values("market value", ascending=False)) #this will sort the dataframe by market value in descending order