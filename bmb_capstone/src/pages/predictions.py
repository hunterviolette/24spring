import pandas as pd
import os
import numpy as np
import plotly.graph_objects as go
import plotly_express as px
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, callback, State, dash_table, register_page

if __name__ == '__main__':
    from util.core import Schmoo
    from util.util import Preprocessing
else:
    from pages.util.core import Schmoo
    from pages.util.util import Preprocessing

register_page(__name__, suppress_callback_exceptions=True)

class Predictions():

  dataPath = './data'
  modelPath = './models'

  def __init__(self) -> None:
    pass
  
  @staticmethod
  def layout():
    return html.Div([

      dbc.Row([
        dbc.Col([
            html.H4("Input Data Directory"),
            dcc.Dropdown(id='dataDirs', options=os.listdir(Predictions.dataPath), multi=False),
        ]),
        dbc.Col([
            html.H4("Models"),
            dcc.Dropdown(id='modelNames', options=os.listdir(Predictions.modelPath), multi=False),
        ]),
        dbc.Col([
            html.H4("Diameter Mean"),
            dcc.Input(id='diamMean', type='number', step='1', value=30),
        ]),
        dbc.Col([
            html.H4("click three times to make predictions"),
            html.Button("click here", id="makePredictions"),
        ]),
      ], align='center'),
      html.Div(id='predictionDiv'),

    ])

  def callbacks(self):
    @callback(
      [Output("dataDirs", 'value'),
      Output("modelNames", 'value'),
      Output("diamMean", 'value'),
      Output('predictionDiv', 'children')
      ],
      Input("makePredictions", "n_clicks"),
      [
        State("dataDirs", "value"),
        State("modelNames", "value"),
        State("diamMean", "value"),
      ],
    )
    def Predict(clicks: int, 
                data_dir: str, 
                model_name: str,
                diam_mean: float,
              ):
      
      def TransparentImage(img, mask, 
                          colorscale: str = 'emrld',
                          colorscale_interp: bool = False
                        ):
        img, transparent_mask = img, mask.astype(float) 
        transparent_mask[transparent_mask == 0] = np.nan

        if colorscale_interp: zmin, zmax = 0, np.max(mask)+10
        else: zmin, zmax = None, None

        return go.Figure(go.Heatmap(z=np.where(np.isnan(transparent_mask), 
                                    img, 
                                    transparent_mask), 
                          zmin=zmin, 
                          zmax=zmax,
                          colorscale=colorscale
                        )).update_layout(
                            margin=dict(l=0.1, r=0.1, t=0.1, b=0.1), 
                            height=450, width=650)

      print(clicks, data_dir, model_name, diam_mean, sep=' ')

      if clicks == None:
        return (None, None, None)
      if clicks == 3: 
        mdiv = []
        res = Schmoo(model_dir=Predictions.modelPath, 
                    data_dir=f"{Predictions.dataPath}/{data_dir}",
                    diam_mean=diam_mean
                  ).TestModel(model_name=model_name, debug=True)
        

        if isinstance(res, list):
          for re in res:
            mdiv.extend([
                html.H3(f"Total cells found: {np.max(re[1])} for image: {re[2]}"),
                dbc.Row([
                    dbc.Col([
                        html.H4("Input image"),
                        dcc.Graph(figure=TransparentImage(re[0], re[1]))
                    ], width=6),
                    dbc.Col([
                        html.H4("Image with predicted mask overlay"),
                        dcc.Graph(figure=TransparentImage(re[0], re[1], 'rdbu', True))

                    ], width=6), 
                ], align='justify'),
            ])
        
        print('plotting...')
        return (data_dir, model_name, diam_mean, mdiv)
      else: 
        return (data_dir, model_name, diam_mean)

x = Predictions()
layout = x.layout()
x.callbacks()