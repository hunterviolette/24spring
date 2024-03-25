import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import pandas as pd

from dash import dcc, html, Input, Output, callback, State, register_page

if __name__ == '__main__':
    from util.util import Preprocessing, DashUtil
else:
    from pages.util.util import Preprocessing, DashUtil

register_page(__name__, suppress_callback_exceptions=True)

class UnitFlows(DashUtil, Preprocessing):

  def __init__(self) -> None:
    pass 

  def layout(self):
    return html.Div([

      dbc.Row([
        dbc.Col([
            html.H4("Max Iterations", style={'text-align': 'center'}),
            dcc.Input(id='ssf_maxIter', type='number', value=15,
                      className=UnitFlows.Formatting('input'),
                      style=UnitFlows.Formatting('textStyle')
                    ),
        ], className='text-center'),
      ], align='center'),

      html.Div(id='ssf_div'),

    ], className='mb-4', style=UnitFlows.Formatting('mdiv'))
  
  def callbacks(self):
    @callback(
          [Output("ssf_maxIter", 'value'),
          Output("ssf_div", 'children'),
          ],
          Input("ssf_maxIter", "value"),
        )
    def cb(maxIter, col:str="Flow (kmol/batch)"):

      UnitFlows.UnitFlowsV2(self)
      df = self.flows
      mdiv = []

      df = df.loc[df["Iteration"] <= maxIter].round(5).sort_values("Iteration")
      for unit in [x for x in df["Unit"].unique()]:
        
        figs = []
        for stream in df.loc[(df["Unit"] == unit)]["Stream Type"].unique():
          d = df.loc[(df["Stream Type"] == stream) & (df["Unit"] == unit)].copy()

          if not d.empty:

            numChems = {chem: i for i, chem in enumerate(d['Chemical'].unique())}
            d['Chemical_numeric'] = d['Chemical'].map(numChems)

            fig = go.Figure()
            for chem, chem_num in numChems.items():
                chem_data = d[d['Chemical'] == chem]
                fig.add_trace(go.Scatter(
                    x=chem_data['Iteration'], 
                    y=chem_data[col], 
                    mode='markers+lines', 
                    marker=dict(color=chem_num),
                    name=chem
                ))
            
            annotations = [
              dict(
                x=x,
                y=y + (d[col].max() - d[col].min()) * 0.0000000005, 
                text=str(y),
                showarrow=False,
                font=dict(size=8),
                yanchor='bottom', 
                yshift=10  
              ) 
              for x, y in zip(d["Iteration"], d[col].round(3))
              if x % 3 == 0
            ]
            
            figs.append(
                fig.update_layout(
                  title=f"Stream: {stream}",
                  xaxis_title='Number of iterations',
                  yaxis_title=col,
                  annotations=annotations,
                  showlegend=True, 
                  legend_title='Compound', 
                  margin=dict(l=0, r=0, t=30, b=0)
                )
              )
            
        if len(figs) == 2:
          mdiv.extend([
              html.H3(f"Unit: {unit}",
                      className=UnitFlows.Formatting()),

              dbc.Row([
                  dbc.Col([
                      dcc.Graph(figure=figs[0]),
                  ], width=6),
                  dbc.Col([
                      dcc.Graph(figure=figs[1]),
                  ], width=6), 

              ], align='justify'),
            ])
        elif len(figs) == 3:
          mdiv.extend([
              html.H3(f"Unit: {unit}",
                      className=UnitFlows.Formatting()),

              dbc.Row([
                  dbc.Col([
                      dcc.Graph(figure=figs[0]),
                  ], width=4),
                  dbc.Col([
                      dcc.Graph(figure=figs[1]),
                  ], width=4), 
                  dbc.Col([
                      dcc.Graph(figure=figs[2]),
                  ], width=4), 

              ], align='justify'),
            ])
              
      return (maxIter, mdiv)
      

x = UnitFlows()
layout = x.layout()
x.callbacks()
