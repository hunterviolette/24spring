import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np

from dash import dcc, html, Input, Output, callback, State, register_page

if __name__ == '__main__':
    from util.util import Preprocessing, DashUtil
else:
    from pages.util.util import Preprocessing, DashUtil

register_page(__name__, suppress_callback_exceptions=True)

class Flows(DashUtil, Preprocessing):

  def __init__(self) -> None:
    pass

  def layout(self):
    Flows.HeatFlows(self) 
    df = self.heat_flows
    col = 'Utility (kW)'

    d = df.copy(deep=True).reset_index(drop=True)
    d[f"Positive {col}"] = d.loc[d[col] > 0, col]
    d[f"Negative {col}"] = d.loc[d[col] < 0, col].__abs__()
    

    sdf = d.fillna(0).groupby("Iteration").agg({
            f"Positive {col}": 'sum',
            f"Negative {col}": 'sum'
      }).reset_index(drop=False)

    sdf_sum = sdf.groupby('Iteration'
                )[[f"Positive {col}", f"Negative {col}"]
                ].sum().reset_index().round(0)

    fig2 = go.Figure()
    for c in ["Positive", "Negative"]:
      fig2.add_trace(
        go.Bar(
          x=sdf['Iteration'],
          y=sdf[f"{c} {col}"],
          text=sdf_sum[f"{c} {col}"],
          textposition='auto',
          name=c
      ))

    numUnit = {chem: i for i, chem in enumerate(df['Unit'].unique())}
    df['unit_numeric'] = df['Unit'].map(numUnit)

    fig = go.Figure()
    for unit, unit_num in numUnit.items():
      unit_data = df[df['Unit'] == unit]
      fig.add_trace(
        go.Scatter(
          x=unit_data['Iteration'], 
          y=unit_data[col], 
          mode='markers+lines', 
          marker=dict(color=unit_num),
          name=unit
      ))
    
    return html.Div([

      Flows.DarkDashTable(self.heat_flows.round(3)),
      
      dcc.Graph(
        figure=fig.update_layout(
                    title=f"Heat Flows by Unit",
                    xaxis_title='Number of iterations',
                    yaxis_title=col,
                    annotations=Flows.FigAnnotations(df, "Iteration", col),
                    showlegend=True, 
                    margin=dict(l=0, r=0, t=30, b=0)
                  )),

      dcc.Graph(
        figure=fig2.update_layout(
          barmode='group',
          title="Positive and Negative Utility by Iteration for Reactors/Electrolyzers",
          xaxis_title="Iteration",
          yaxis_title="Utility (kW)"
      ))

    ], className='mb-4', style=Flows.Formatting('mdiv'))
  
  def callbacks(self):
    pass

x = Flows()
layout = x.layout()
x.callbacks()
