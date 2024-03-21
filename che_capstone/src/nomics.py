import pandas as pd
import plotly_express as px
from math import log2

cost = 100

df = pd.DataFrame()
for x in range(1, 1000, 1):
  if x == 1: kn = cost
  else: kn = cost * x**log2(.8)

  df = pd.concat([df, 
          pd.DataFrame({
              "Module": [x],
              "Cost": [kn]
            })
        ])
  
print(df) 

px.line(df, x="Module", y="Cost").show()