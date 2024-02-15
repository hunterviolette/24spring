import numpy as np
import plotly.graph_objects as go
import plotly_express as px
import dash_bootstrap_components as dbc
import pandas as pd
import os

from dash import dcc, html, Input, Output, callback, State, dash_table, register_page
from typing import Optional

if __name__ == '__main__':
    from util.core import Schmoo
    from util.util import Preprocessing, DashUtil
else:
    from pages.util.core import Schmoo
    from pages.util.util import Preprocessing, DashUtil

register_page(__name__, 
              prevent_initial_callbacks=True,
              suppress_callback_exceptions=True
            )

class Predictions(DashUtil):

  dataPath = './vol/image_data'
  modelPath = './vol/models'
  predictPath = './vol/predictions'

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
                      ),
        ]),
        dbc.Col([
            html.H4("Models", style={'text-align': 'center'}),
            html.Br(),
            dcc.Dropdown(id='modelNames', multi=False,
                        style=Predictions.Formatting('textStyle'),
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
            html.H4("Image pixel resize", style={'text-align': 'center'}),
            dcc.Input(id='resizeImage', type='number', value=450,
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
      [Output("dataDirs", "options"),
      Output("modelNames", 'options')],
      Input("makePredictions", "n_clicks"),
    )
    def initPred(clicks):
      if clicks == 0:
        return (
          os.listdir(Predictions.dataPath),
          os.listdir(Predictions.modelPath)
        )


    @callback(
      [Output("dataDirs", 'value'),
      Output("modelNames", 'value'),
      Output("diamMean", 'value'),
      Output("numPredictions", 'value'),
      Output("saveImage", 'value'),
      Output('resizeImage', 'value'),
      Output('predictionDiv', 'children'),
      ],
      Input("makePredictions", "n_clicks"),
      [State("dataDirs", "value"),
      State("modelNames", "value"),
      State("diamMean", "value"),
      State("numPredictions", "value"),
      State("saveImage", "value"),
      State("resizeImage", "value"),
      ],
    )
    def Predict(clicks: int, 
                data_dir: str, 
                model_name: str,
                diam_mean: float,
                numPredictions: Optional[int],
                saveImage: bool,
                resizeImage: Optional[int],
              ):
      
      print(clicks, data_dir, model_name, diam_mean, 
            numPredictions, saveImage, resizeImage, 
            sep=' ')
      
      mdiv = []
      if clicks > 0: 
        res = Schmoo(model_dir=Predictions.modelPath, 
                    data_dir=f"{Predictions.dataPath}/{data_dir}",
                    predict_dir=Predictions.predictPath,
                    diam_mean=diam_mean
                  ).Predict(
                          model_name=model_name, 
                          numPredictions=numPredictions,
                          saveImages=saveImage,
                          imgResize=resizeImage,
                          hasTruth=False,
                        )
        
        if isinstance(res, list):
          for re in res:
            mdiv.extend([
                html.H3(f"Total cells found: {np.max(re[1])} for image: {re[2]}",
                        className=Predictions.Formatting()),

                dbc.Row([
                    dbc.Col([
                        html.H4("Input image"),
                        Predictions.PlotImage(re[0])
                    ], width=6),
                    dbc.Col([
                        html.H4("Image with predicted mask overlay"),
                        Predictions.TransparentImage(re[0], re[1], 'ice', True),
                    ], width=6), 
                ], align='justify'),
            ])
        
        print('plotting...')
      return (data_dir, model_name, diam_mean, 
              numPredictions, saveImage, resizeImage, mdiv
            )

x = Predictions()
layout = x.layout()
x.callbacks()