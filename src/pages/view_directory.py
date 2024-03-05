import numpy as np
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import os

from dash import dcc, html, Input, Output, callback, State, dash_table, register_page
from typing import Optional
from cv2 import resize

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

class ViewDir(DashUtil, Preprocessing):

  dataPath = './vol/image_data'

  def __init__(self) -> None:
    pass
  
  @staticmethod
  def layout():
    return html.Div([

      dbc.Row([
        dbc.Col([
            html.H4("Input directory", style={'text-align': 'center'}),
            dcc.Dropdown(id='view_dataDirs', multi=False,
                        style=ViewDir.Formatting('textStyle'),
                      ),
        ]),
        dbc.Col([
            html.H4("Max Images", style={'text-align': 'center'}),
            dcc.Input(id='view_numPredictions', type='number', value=None,
                      className=ViewDir.Formatting('input'),
                      style=ViewDir.Formatting('textStyle')
                    ),
        ]),
        dbc.Col([
            html.H4("Image resize", style={'text-align': 'center'}),
            dcc.Input(id='view_resizeImage', type='number', value=600,
                      className=ViewDir.Formatting('input'),
                      style=ViewDir.Formatting('textStyle')
                    ),
        ]),
        dbc.Col([
            html.H4("Click for model predictions"),
            html.Button("click here", n_clicks=0, id="view_makePredictions", 
                        className=ViewDir.Formatting('button', 'info')
                      ),
        ], className='text-center'),
      ], align='center'),
      html.Div(id='view_predictionDiv'),

    ], className='mb-4', style=ViewDir.Formatting('mdiv'))

  def callbacks(self):
    @callback(
      [Output("view_dataDirs", "options")],
      Input("view_makePredictions", "n_clicks"),
    )
    def initPred(clicks):
      return (
        os.listdir(ViewDir.dataPath),
      )

    @callback(
      [Output("view_dataDirs", 'value'),
      Output("view_numPredictions", 'value'),
      Output('view_resizeImage', 'value'),
      Output('view_predictionDiv', 'children'),
      ],
      Input("view_makePredictions", "n_clicks"),
      [State("view_dataDirs", "value"),
      State("view_numPredictions", "value"),
      State("view_resizeImage", "value"),
      ],
    )
    def Predict(clicks: int, 
                data_dir: str, 
                numPredictions: Optional[int],
                resizeImage: Optional[int],
              ):
      
      print(clicks, data_dir, 
            numPredictions, resizeImage,
            sep=' ')

      mdiv = []
      mdiv.append(html.H2(ViewDir.TorchGPU(), 
                  className=ViewDir.Formatting(color='warning')))
      
      if clicks > 0 and data_dir != None:

        if "_nomask" in data_dir: hasMask = False
        else: hasMask = True


        res = Schmoo(data_dir=f"{ViewDir.dataPath}/{data_dir}",
                  ).DataGenerator(maskRequired=hasMask)
        
        if isinstance(res, dict):
          for i, key in enumerate(res.keys()):
            row = res[key]
            if isinstance(resizeImage, int):
              rTup = (resizeImage, resizeImage)
              img = resize(row["img"], rTup)
              if hasMask: mask = resize(row["mask"], rTup)
            else:
              img = row["img"]
              if hasMask: mask = row["mask"]
            
            if hasMask:
              mdiv.extend([
                html.H3(f"Image/Mask filename: {key}",
                        className=ViewDir.Formatting()),

                dbc.Row([
                    dbc.Col([
                        html.H4("True mask overlay"),
                        ViewDir.TI2(img, mask, wh=resizeImage, zoom=True),
                    ], width=6),
                    dbc.Col([
                        html.H4("Input image"),
                        ViewDir.PI2(img, wh=resizeImage, zoom=True),
                    ], width=6), 

                ], align='justify'),
              ])
            
            else:
              mdiv.extend([
                html.H3(f"Image filename: {key}",
                        className=ViewDir.Formatting()),
                
                ViewDir.PI2(img, wh=resizeImage, zoom=True),
              ])
            
            if isinstance(numPredictions, int):
              if i+1 >= numPredictions: break

        
        print('plotting...')
      else:
        rules = dcc.Markdown('''
            ## What does clicking this button do? 
              - Plot images in a directory within `vol/image_data`
                - If images has mask pairs, will also overlay segmented mask overlay on input image

            ## Input variables
            1. Input directory - *_Required field_*
                - Each `directory` of images in `vol/image_data`.
                - Add sets of images using the [Upload page](http://localhost:8050/upload)

            5. Max Images - Default:None
                ```
                If max images == None: render all images on webpage
                else: render first x images in directory
                ```

            6. Image resize - Default:450
                ```
                Resizes the image and mask for quicker rendering
                Can clear input for no resize
                ```
            ''', 
          style={
              'backgroundColor': '#121212',
              'color': '#FFFFFF',       
              'padding': '20px',     
            }
          )

        mdiv.append(rules)
      return (data_dir, numPredictions, resizeImage, mdiv)

x = ViewDir()
layout = x.layout()
x.callbacks()