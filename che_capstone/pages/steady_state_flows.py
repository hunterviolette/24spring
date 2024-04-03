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
            html.H4("Basis", style={'text-align': 'center'}),
            dcc.Dropdown(id='ssf_basis', multi=False,
                        style=UnitFlows.Formatting('textStyle'),
                        options=['Mole', 'Mass'], value='Mole'
                      ),
        ], className='text-center'),
      ], align='center'),

      html.Div(id='ssf_div'),

    ], className='mb-4', style=UnitFlows.Formatting('mdiv'))
  
  def callbacks(self):
    @callback(
          [Output("ssf_basis", 'value'),
          Output("ssf_div", 'children'),
          ],
          Input("ssf_basis", "value"),
        )
    def cb(basis, col:str="Flow (kmol/batch)"):
      
      if basis == "Mole": col = "Flow (kmol/batch)"
      else: col = "Flow (kg/batch)"

      UnitFlows.UnitFlowsV2(self)
      df = self.flows.round(5).sort_values("Iteration")
      mdiv = []

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
            
            
            figs.append(
                fig.update_layout(
                  title=f"Stream: {stream}",
                  xaxis_title='Number of iterations',
                  yaxis_title=col,
                  annotations=UnitFlows.FigAnnotations(d, "Iteration", col),
                  showlegend=True, 
                  legend_title='Compound', 
                  margin=dict(l=0, r=0, t=30, b=0)
                )
              )
            
        mdiv.extend([
          html.H3(f"Unit: {unit}", className=UnitFlows.Formatting()),
          dbc.Row(
              [
                dbc.Col(
                    [dcc.Graph(figure=figs[i])],
                    width=12 // len(figs)
                ) for i in range(len(figs))
              ],
              align='justify'
            )
        ])
                
      return (basis, mdiv)
      

x = UnitFlows()
layout = x.layout()
x.callbacks()
