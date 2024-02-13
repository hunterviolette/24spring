import numpy as np
import plotly.graph_objects as go
import plotly_express as px
import dash_bootstrap_components as dbc
import pandas as pd
import io
import base64
import os

from dash import dcc, html, Input, Output, callback, State, dash_table, register_page
from typing import Optional
from tifffile import imread

if __name__ == '__main__':
    from util.core import Schmoo
    from util.util import Preprocessing, DashUtil
else:
    from pages.util.core import Schmoo
    from pages.util.util import Preprocessing, DashUtil

register_page(__name__, suppress_callback_exceptions=True)

class Upload(DashUtil):

  def __init__(self) -> None:
    pass
  
  @staticmethod
  def layout():
    return html.Div([
            html.Button("click here", n_clicks=0, id="uploadButton", 
                              className=Upload.Formatting('button', 'info')
                            ),
            html.Div(id='upload_mdiv')
          ])
  
  def callbacks(self):
    @callback(
        Output('upload_mdiv', 'children'),
        [Input('uploadButton', 'n_clicks')],
        prevent_initial_call=True
    )
    def update_output(clicks):
      if clicks > 0: 
        fileList = os.listdir('./load_images')
        for file in fileList:
          print(file)

        return html.Div('Files in load_images:' + str(fileList))
      else: return html.Div("Click the button to upload .tif and .png files in load_images folder")
        
x = Upload()
layout = x.layout()
x.callbacks()
