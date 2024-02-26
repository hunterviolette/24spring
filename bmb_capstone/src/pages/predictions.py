import numpy as np
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
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

class Predictions(DashUtil, Preprocessing):

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
            dcc.Dropdown(id='dataDirs', multi=False,
                        style=Predictions.Formatting('textStyle'),
                      ),
        ]),
        dbc.Col([
            html.H4("Models", style={'text-align': 'center'}),
            dcc.Dropdown(id='modelNames', multi=False,
                        style=Predictions.Formatting('textStyle'),
                      ),
        ]),
        dbc.Col([
            html.H4("Save Masks", style={'text-align': 'center'}),
            dcc.Dropdown(id='saveImage', multi=False, value=False,
                              style=Predictions.Formatting('textStyle'),
                              options=[
                                  {'label': 'True', 'value': True},
                                  {'label': 'False', 'value': False}
                                ]
                            ),
        ]),
        dbc.Col([
            html.H4("Diameter mean", style={'text-align': 'center'}),
            dcc.Input(id='diamMean', type='number', step='1', value=30, 
                      className=Predictions.Formatting('input'),
                      style=Predictions.Formatting('textStyle')
                    ),
        ]),
        dbc.Col([
            html.H4("Max predictions", style={'text-align': 'center'}),
            dcc.Input(id='numPredictions', type='number', value=None,
                      className=Predictions.Formatting('input'),
                      style=Predictions.Formatting('textStyle')
                    ),
        ]),
        dbc.Col([
            html.H4("Image resize", style={'text-align': 'center'}),
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
                diam_mean: int,
                numPredictions: Optional[int],
                saveImage: bool,
                resizeImage: Optional[int],
              ):
      
      print(clicks, data_dir, model_name, diam_mean, 
            numPredictions, saveImage, resizeImage,
            sep=' ')

      mdiv, ajiList = [], []
      mdiv.append(html.H2(Predictions.TorchGPU(), 
                  className=Predictions.Formatting(color='warning')))
      
      if clicks > 0 and \
          data_dir != None and \
          model_name != None and \
          diam_mean != None:

        if "_nomask" in data_dir: hasMask = False
        else: hasMask = True

        res = Schmoo(model_dir=Predictions.modelPath, 
                    data_dir=f"{Predictions.dataPath}/{data_dir}",
                    predict_dir=Predictions.predictPath,
                  ).Predict(
                          model_name=model_name, 
                          numPredictions=numPredictions,
                          saveImages=saveImage,
                          imgResize=resizeImage,
                          hasTruth=hasMask,
                          diamMean=diam_mean
                        )
        
        if isinstance(res, list):
          for re in res:
            if hasMask and isinstance(re[3], float): 
              aji = f" with Jaccard index value: {round(re[3], 4)}"
              ajiList.append(re[3])

              mdiv.extend([
                  html.H3(f"Found {np.max(re[1])} cells for image {re[2]}{aji}",
                          className=Predictions.Formatting()),

                  dbc.Row([
                      dbc.Col([
                          html.H4("True mask overlay"),
                          Predictions.TI2(re[0], re[4]),
                      ], width=4),
                      dbc.Col([
                          html.H4("Input image"),
                          Predictions.PlotImage(re[0]),
                      ], width=4), 
                      dbc.Col([
                          html.H4("Predicted mask overlay"),
                          Predictions.TI2(re[0], re[1]),
                      ], width=4), 

                  ], align='justify'),
                ])
              
            else: 
              aji = ""

              mdiv.extend([
                  html.H3(f"Found {np.max(re[1])} cells for image {re[2]}{aji}",
                          className=Predictions.Formatting()),

                  dbc.Row([
                      dbc.Col([
                          html.H4("Input image"),
                          Predictions.PlotImage(re[0], w=650),
                      ], width=6),
                      dbc.Col([
                          html.H4("Predicted mask overlay"),
                          Predictions.TI2(re[0], re[1], w=650),
                      ], width=6), 
                  ], align='justify'),
                ])

                
          if hasMask and len(ajiList) >0: 
            mdiv.insert(1, # append to 1 index
                        html.H4(f"Mean Jaccard index: {round(sum(ajiList)/len(ajiList),4)}", 
                                className=Predictions.Formatting(color='primary')
                        )
                      )
            mdiv.insert(1, # append to 1 index
                        html.H2(f"{[round(x, 4) for x in ajiList]}", 
                                className=Predictions.Formatting(color='primary')
                        )
                      )
        
        print('plotting...')
      else:
        rules = dcc.Markdown('''
            1. Input directory
                ```
                - Each directory of images in vol/image_data.
                - Add sets of images as input directories using the Upload page 
                - Image(s)/mask(s) must be: 2-dimensional and (png or tif)
                ```

            2. Models
                ```
                The cellpose models in vol/models
                ```

            3. Save predictions
                ```
                Saves predicted mask to vol/predictions
                ```

            4. Diameter mean
                ```
                The estimated pixel diameter of the cells
                Try 30, 80, 15, 120 if you are having 
                issues with meaningful predictions
                ```

            5. Max predictions
                ```
                If max predictions == None: predict all images in folder
                else: predict first x images in directory
                ```

            6. Image resize
                ```
                Resizes the image and mask for quicker rendering
                Default resize is (450, 450)
                ```
            ''', 
          style={
              'backgroundColor': '#121212',
              'color': '#FFFFFF',       
              'padding': '20px',     
            }
          )

        mdiv.append(rules)
      return (data_dir, model_name, diam_mean, numPredictions, 
              saveImage, resizeImage, mdiv
            )

x = Predictions()
layout = x.layout()
x.callbacks()