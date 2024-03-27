import pandas as pd
import plotly_express as px
import numpy as np
import os

if os.getcwd().split("/")[-1].endswith("src"):
  from unit_registry import UnitConversion
else: 
  from src.unit_registry import UnitConversion

class Nomics(UnitConversion):
  def __init__(self) -> None:
    super().__init__()


  def main():

    cost = 10900000
    years = 20 # plant life
    numModules = 1000

    modules = np.arange(1, numModules + 1)
    costs = np.where(modules == 1, cost, cost * modules ** np.log2(0.8))

    df = pd.DataFrame({"Modules": abs(modules), "Cost": costs})

    df["Fixed Cost"] = df["Cost"].cumsum()
    df["20-Year Annualized Fixed Cost ($)"] = df["Fixed Cost"] / years
    
    df["Module Rev"] = 800*2*360
    df["Module Exp"] = 126864 + 50250
    
    df["Cost Scale"] = df["Fixed Cost"] / df["Fixed Cost"].min()
    df["Free Money Scalar"] = df["Modules"] / df["Cost Scale"]

    df["Revenue"] = df["Module Rev"] * df["Modules"]
    df["Expenses"] = df["Module Exp"] * df["Modules"]
    df["Annualized Profit ($)"] = df["Revenue"] - df["Expenses"] - df["20-Year Annualized Fixed Cost ($)"]

    if numModules > 10000: df = df.iloc[::len(df) // 1000]

    px.line(df, x="Modules", y=["20-Year Annualized Fixed Cost ($)", 
                                "Annualized Profit ($)"]
              ).update_layout(yaxis_title="Value").show()
    
    fig = px.line(df, x="Modules", y="Cost"
                ).update_layout(yaxis_title="Cost per unit ($)")

    for i, row in df.iterrows():
      if i % 50 == 0:  
        cost_text = f"{row['Cost']:.2e}"[:4]
        fig.add_annotation(x=row["Modules"] + 5, y=row["Cost"], text=cost_text,
                          showarrow=False, font=dict(size=8)) 

    fig.show()

if __name__ == "__main__":
  Nomics.main()