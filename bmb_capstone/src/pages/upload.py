import numpy as np
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import os

from dash import dcc, html, Input, Output, callback, State, dash_table, register_page
from tifffile import imwrite

if __name__ == '__main__':
    from util.util import Preprocessing, DashUtil
else:
    from pages.util.util import Preprocessing, DashUtil

register_page(__name__, suppress_callback_exceptions=True)

class Upload(DashUtil):

  load_dir = './vol/image_loader'
  image_dir = './vol/image_data'

  def __init__(self) -> None:
    pass
  
  @staticmethod
  def layout():
    return html.Div([
            dbc.Row([
              dbc.Col([
                  html.H4("Image Set Tag", style={'text-align': 'center'}),
                  html.Br(),
                  dcc.Input(id='uploadTag', type='text', value=None,
                            className=Upload.Formatting('input'),
                            style=Upload.Formatting('textStyle')
                          ),
              ]),
              dbc.Col([
                  html.H4("Has mask file", style={'text-align': 'center'}),
                  html.Br(),
                  dcc.Dropdown(id='uploadMask', multi=False,
                              style=Upload.Formatting('textStyle'),
                              options=[
                                  {'label': 'True', 'value': True},
                                  {'label': 'False', 'value': False}
                                ]
                            ),
              ]),
              dbc.Col([
                  html.H4("Mask file is segmented", style={'text-align': 'center'}),
                  html.Br(),
                  dcc.Dropdown(id='uploadSeg', multi=False,
                      style=Upload.Formatting('textStyle'),
                      options=[
                          {'label': 'True', 'value': True},
                          {'label': 'False', 'value': False}
                        ]
                      ),
              ]),
              dbc.Col([
                  html.H4("Click to load images"),
                  html.Button("click here", n_clicks=0, id="uploadButton", 
                              className=Upload.Formatting('button', 'info')
                            ),
              ], className='text-center'),

            ], align='center'),
            html.Div(id='upload_mdiv'),

          ], className='mb-4', style=Upload.Formatting('mdiv'))  
  
  def callbacks(self):
    @callback(
      [Output('upload_mdiv', 'children'),
      Output("uploadTag", 'value'),
      Output("uploadMask", 'value'),
      Output("uploadSeg", 'value'),
      ],
      Input('uploadButton', 'n_clicks'),
      [State("uploadTag", "value"),
      State("uploadMask", "value"),
      State("uploadSeg", "value"),
      ],
    )
    def update_output(clicks, tag, hasMask, segmented):
      print(clicks, tag, hasMask, segmented, sep=', ')
      if clicks > 0 and tag != None and hasMask != None and segmented != None: 
        fileList = os.listdir(Upload.load_dir)
        print(fileList)

        mdiv, reject = [], []
        x = Preprocessing(Upload.load_dir, Upload.image_dir)
        filefinder = {x.NameCleaner(f):f for f in fileList}
        
        fileDict = {}
        for file in [f for f in fileList if f.endswith(('.tif', '.png'))]:
          try:
            name = str(x.NameCleaner(file))
            print(name)

            if "mask" not in name:
              img = x.ReadImage(file, Upload.load_dir)
              fileDict[name] = {"img": img, "mask": None, "filename": file}
          except:
            reject.append(f"file {file} was unable to be opened as .tif or .png")
        
        print(fileDict.keys(), len(fileDict.keys()))

        if hasMask and len(fileDict.keys()) > 1:
          for key in fileDict.keys():
            print(f"this key is: {key}")
            if ".tif" in key: tif, png = True, False
            elif ".png" in key: png, tif = True, False
            maskName = key.replace(".", "_mask.")
            print(f"mask name: {maskName}")
            print(filefinder[maskName])

            if maskName in filefinder:
              mask = x.ReadImage(filefinder[maskName], Upload.load_dir)
              if not segmented: mask = x.SegmentMask(mask) # Hand-drawn imageJ masks
              fileDict[key]["mask"] = mask
            else:
              if tif: name = maskName.replace(".tif", ".png")
              elif png: name = maskName.replace(".png", ".tif")

              if name in filefinder: 
                mask = x.ReadImage(file, Upload.load_dir)
                if not segmented: mask = x.SegmentMask(mask) # Hand-drawn imageJ masks
                fileDict[key]["mask"] = mask
              else: reject.append(
                            f"Could not find a mask file for "
                            f"{fileDict[key]['filename']}, expected filename: "
                            f"{maskName.split('.')[0]} and be .png or .tif"
                          )
        
        for f in fileDict.keys():
          print(fileDict[f]["mask"])

        if len(fileDict.keys()) > 1: 
          popKeys = []
          for key in fileDict.keys():
            row = fileDict[key]
            image, mask = row["img"], row["mask"]
            
            for f in [[image, "Image"], [mask, "Mask"]]:

              if isinstance(f[0], np.ndarray):
                if len(f[0].shape) != 2:
                  reject.append(f"{f[1]} has shape of {len(f[0].shape)}, expected 2")
                  popKeys.append(key)
              else:
                reject.append(f"{f[1]} could not be opened: {key}")
                popKeys.append(key)
        
          if len(popKeys) > 0:
            for key in list(set(popKeys)): fileDict.pop(key)

        
        if len(fileDict.keys()) > 1: 
          x.initDir(f"{Upload.image_dir}/{tag}")

          for key in fileDict.keys():
            imwrite(key, fileDict[key]["img"])
            imwrite(key.replace('.', '_mask.'), fileDict[key]["mask"])
          
          for file in fileList:
            os.remove(f"{Upload.load_dir}/{file}")

        if hasMask: mdiv.append(html.H2("Accepted image/mask pairs:"))
        else: mdiv.append(html.H2("Accepted images:"))

        for key in fileDict.keys():mdiv.append(html.H5(key))

        if hasMask: mdiv.append(html.H2("Rejcted image/mask pairs:"))
        else: mdiv.append(html.H2("Rejected images:"))

        if len(reject) == 0: mdiv.append(html.H5("No rejected images!"))
        else: 
          for r in reject: mdiv.append(html.H2(r))

        return (mdiv, tag, hasMask, segmented)
      else: 
        msg = " ".join([
                    "To start, place images in vol/load_images"
                    "and enter a unique tag for set of images in load_images"
                    ])
        return (html.H2(msg), tag, hasMask, segmented)
        
x = Upload()
layout = x.layout()
x.callbacks()
