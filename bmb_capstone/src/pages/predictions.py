import pandas as pd
import os
import numpy as np
import plotly.graph_objects as go
import plotly_express as px
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, callback, State, dash_table, register_page
from typing import Optional

if __name__ == '__main__':
    from util.core import Schmoo
    from util.util import Preprocessing
else:
    from pages.util.core import Schmoo
    from pages.util.util import Preprocessing

register_page(__name__, suppress_callback_exceptions=True)

class DashUtil:
  @staticmethod
  def Formatting( className: str = 'heading', 
                      color: str = 'info',
                      textAlign: str = 'center'
                    ):

    if className == 'heading':
      return f"bg-opacity-50 p-1 m-1 bg-{color} text-dark fw-bold rounded text-{textAlign}"
    
    elif className == 'mdiv': return {"padding": "10px"} # style not className

    elif className == 'button': return f"btn btn-{color}"

    elif className == 'input': return 'form-control'

    elif className == 'textStyle': return {'text-align': 'center', 'color': 'black'} # style
    
    else: raise Exception("className not found")

class Predictions(DashUtil):

  dataPath = './data'
  modelPath = './models'

  def __init__(self) -> None:
    pass
  
  @staticmethod
  def layout():
    return html.Div([

      dbc.Row([
        dbc.Col([
            html.H4("Input directory", style={'text-align': 'center'}),
            html.Br(),
            dcc.Dropdown(id='dataDirs', multi=False,
                        style=Predictions.Formatting('textStyle'),
                        options=os.listdir(Predictions.dataPath)
                      ),
        ]),
        dbc.Col([
            html.H4("Models", style={'text-align': 'center'}),
            html.Br(),
            dcc.Dropdown(id='modelNames', multi=False,
                        style=Predictions.Formatting('textStyle'),
                        options=os.listdir(Predictions.modelPath)
                      ),
        ]),
        dbc.Col([
            html.H4("Save Images", style={'text-align': 'center'}),
            html.Br(),
            dcc.Dropdown(id='saveImage', multi=False,
                style=Predictions.Formatting('textStyle'),
                options=[
                    {'label': 'True', 'value': True},
                    {'label': 'False', 'value': False}
                  ]
                ),
        ]),
        dbc.Col([
            html.H4("Diameter mean", style={'text-align': 'center'}),
            html.Br(),
            dcc.Input(id='diamMean', type='number', step='1', value=30, 
                      className=Predictions.Formatting('input'),
                      style=Predictions.Formatting('textStyle')
                    ),
        ]),
        dbc.Col([
            html.H4("Max number of predictions", style={'text-align': 'center'}),
            dcc.Input(id='numPredictions', type='number', value=None,
                      className=Predictions.Formatting('input'),
                      style=Predictions.Formatting('textStyle')
                    ),
        ]),
        dbc.Col([
            html.H4("Click for model predictions"),
            html.Button("click here", n_clicks=0, id="makePredictions", 
                        className=Predictions.Formatting('button', 'info')
                      ),
        ], className='text-center'),
      ], align='center'),
      html.Div(id='predictionDiv'),

    ], className='mb-4', style=Predictions.Formatting('mdiv'))

  def callbacks(self):
    @callback(
      [Output("dataDirs", 'value'),
      Output("modelNames", 'value'),
      Output("diamMean", 'value'),
      Output("numPredictions", 'value'),
      Output("saveImage", 'value'),
      Output('predictionDiv', 'children')
      ],
      Input("makePredictions", "n_clicks"),
      [
        State("dataDirs", "value"),
        State("modelNames", "value"),
        State("diamMean", "value"),
        State("numPredictions", "value"),
        State("saveImage", "value"),
      ],
    )
    def Predict(clicks: int, 
                data_dir: str, 
                model_name: str,
                diam_mean: float,
                numPredictions: Optional[int],
                saveImage: bool,
              ):
      
      def TransparentImage(img, mask, 
                          colorscale: str = 'emrld',
                          colorscale_interp: bool = False
                        ):
        img, transparent_mask = img, mask.astype(float) 
        transparent_mask[transparent_mask == 0] = np.nan

        if colorscale_interp: zmin, zmax = 0, np.max(mask)+1
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

      print(clicks, data_dir, model_name, diam_mean, numPredictions, saveImage, sep=' ')
      mdiv = []

      if clicks > 0: 
        res = Schmoo(model_dir=Predictions.modelPath, 
                    data_dir=f"{Predictions.dataPath}/{data_dir}",
                    diam_mean=diam_mean
                  ).TestModel(model_name=model_name, 
                              numPredictions=numPredictions,
                              saveImages=saveImage
                            )
        
        if isinstance(res, list):
          for re in res:
            mdiv.extend([
                html.H3(f"Total cells found: {np.max(re[1])} for image: {re[2]}",
                        className=Predictions.Formatting()),
                dbc.Row([
                    dbc.Col([
                        html.H4("Input image"),
                        dcc.Graph(figure=go.Figure(go.Heatmap(
                          z=re[0], colorscale='greys'
                          )).update_layout(
                              margin=dict(l=0.1, r=0.1, t=0.1, b=0.1), 
                              height=450, width=650
                        ))
                    ], width=6),
                    dbc.Col([
                        html.H4("Image with predicted mask overlay"),
                        dcc.Graph(figure=TransparentImage(re[0], re[1], 'ice', True))

                    ], width=6), 
                ], align='justify'),
            ])
        
        print('plotting...')
      return (data_dir, model_name, diam_mean, numPredictions, saveImage, mdiv)

x = Predictions()
layout = x.layout()
x.callbacks()