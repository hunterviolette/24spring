import pandas as pd
import plotly_express as px
import numpy as np

cost = 100
prodX = 170000000 # Worlds supply of Ammonia per year

modules = np.arange(1, prodX + 1)
costs = np.where(modules == 1, cost, cost * modules ** np.log2(0.8))

df = pd.DataFrame({"Modules": modules, "Cost": costs})

df["Total Cost"] = df["Cost"].cumsum()
df["Cost Scale"] = df["Total Cost"] / df["Total Cost"].min()
df["Free Money Scalar"] = df["Modules"] / df["Cost Scale"]

df = df.iloc[::len(df) // 1000]

print(df)
px.line(df, x="Modules", y="Free Money Scalar").show()