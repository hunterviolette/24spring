import pandas as pd
import plotly_express as px
from math import log2

cost = 3200000
prodX = 100000

df = pd.DataFrame()
for x in range(1, prodX, 1):
  if x == 1: kn = cost
  else: kn = cost * x**log2(.8)

  df = pd.concat([df, 
          pd.DataFrame({
              "Module": [x],
              "Cost": [kn]
            })
        ])

df["Total Cost"] = df["Cost"].cumsum()

costX = df["Total Cost"].max() / df["Total Cost"].min()

print(prodX / costX, prodX, costX, sep='\n')
#px.line(df, x="Module", y="Total Cost").show()