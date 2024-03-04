import numpy as np
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import os
import shutil

from dash import dcc, html, Input, Output, callback, State, dash_table, register_page
from tifffile import imwrite
from skimage.io import imread as pngRead
from cv2 import resize

if __name__ == '__main__':
    from util.util import Preprocessing, DashUtil
else:
    from pages.util.util import Preprocessing, DashUtil

register_page(__name__, suppress_callback_exceptions=True)

class Upload(DashUtil, Preprocessing):

  load_dir = './vol/image_loader'
  image_dir = './vol/image_data'

  def __init__(self) -> None:
    pass
  
  @staticmethod
  def layout():
    return html.Div([
            dbc.Row([
              dbc.Col([
                  html.H4("Set tag", style={'text-align': 'center'}),
                  html.Br(),
                  dcc.Input(id='uploadTag', type='text', value=None,
                            className=Upload.Formatting('input'),
                            style=Upload.Formatting('textStyle')
                          ),
              ]),
              dbc.Col([
                  html.H4("Has mask", style={'text-align': 'center'}),
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
                  html.H4("Complete validation", style={'text-align': 'center'}),
                  html.Br(),
                  dcc.Dropdown(id='uploadVal', multi=False,
                      style=Upload.Formatting('textStyle'),
                      options=[
                          {'label': 'True', 'value': True},
                          {'label': 'False', 'value': False}
                        ],
                      value=True
                      ),
              ]),
              dbc.Col([
                  html.H4("Non-segmented imageJ masks", style={'text-align': 'center'}),
                  dcc.Dropdown(id='uploadimgJ', multi=False,
                      style=Upload.Formatting('textStyle'),
                      options=[
                          {'label': 'True', 'value': True},
                          {'label': 'False', 'value': False}
                        ],
                      value=False
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
      Output("uploadVal", 'value'),
      Output("uploadimgJ", 'value'),
      ],
      Input('uploadButton', 'n_clicks'),
      [State("uploadTag", "value"),
      State("uploadMask", "value"),
      State("uploadVal", 'value'),
      State("uploadimgJ", "value"),
      ],
    )
    def update_output(clicks, tag, hasMask, val, imageJ):
      print(clicks, tag, hasMask, val, sep=', ')
      mdiv = []
      mdiv.append(html.H2(Upload.TorchGPU(), 
                  className=Upload.Formatting(color='warning')))
      
      if clicks > 0 and \
          tag != None and \
          hasMask != None and \
          val != None: 

        Upload.VsiToTif(Upload.load_dir, Upload.load_dir)

        if not hasMask: tag = f"{tag}_nomask"

        fileList = [f for f in os.listdir(Upload.load_dir) 
                    if f.endswith(('.tif', '.png'))]
        
        if val: fullVal = len(fileList)

        reject = []
        filefinder = {Upload.NameCleaner(f):f for f in fileList}
        
        fileDict = {}
        for file in fileList:
          try:
            name = str(Upload.NameCleaner(file))

            if "mask" not in name:
              img = Upload.ReadImage(file, Upload.load_dir)
              fileDict[name] = {"img": img, "mask": None, "filename": file}
          except:
            reject.append(f"file {file} was unable to be opened as .tif or .png")
        
        if hasMask and len(fileDict.keys()) > 1:
          for key in fileDict.keys():
            if ".tif" in key: tif, png = True, False
            elif ".png" in key: png, tif = True, False
            maskName = key.replace(".", "_mask.")

            if maskName in filefinder:
              ''' This is not working for cellpose segmented masks
              mask = Upload.SegmentMask( # accept both deseg and seg 
                        Upload.DesegmentMask( # so always return a seg mask
                          Upload.ReadImage(filefinder[maskName], Upload.load_dir)))
              '''
              mask = Upload.ReadImage(filefinder[maskName], Upload.load_dir)
              if imageJ: mask = Upload.SegmentMask(mask)

              fileDict[key]["mask"] = mask
            else:
              if tif: name = maskName.replace(".tif", ".png")
              elif png: name = maskName.replace(".png", ".tif")

              if name in filefinder: 
                mask = Upload.ReadImage(filefinder[name], Upload.load_dir)
                if imageJ: mask = Upload.SegmentMask(mask)
                
                fileDict[key]["mask"] = mask
              else: reject.append(
                            f"Could not find a mask file for "
                            f"{key}, expected filename: "
                            f"{maskName.split('.')[0]} and be .png or .tif"
                          )

        if len(fileDict.keys()) > 1: 
          popKeys = []
          for key in fileDict.keys():
            row = fileDict[key]
            image, mask = row["img"], row["mask"]
            
            if hasMask: files = [[image, "Image"], [mask, "Mask"]]
            else: files = [[image, "Image"]]

            for f in files:
              if isinstance(f[0], np.ndarray):
                if len(f[0].shape) != 2:
                  reject.append(f"{f[1]}: {key} had {len(f[0].shape)} dimensions, expected 2")
                  popKeys.append(key)
              else:
                reject.append(f"{f[1]} could not be opened: {key}")
                popKeys.append(key)
        
          if len(popKeys) > 0:
            print(f"{popKeys} removed from dataGen")
            for key in list(set(popKeys)): fileDict.pop(key)
        
        def WriteWrap():
          if hasMask: 
            if tag + "_nomask" in [
                      x for x in os.listdir(Upload.image_dir)
                      if os.path.isdir(f"{Upload.image_dir}/{x}")
                    
                    ]: shutil.rmtree(f"{Upload.image_dir}/{tag}_nomask")

          mdiv.append(html.H2(f"Writing directory {tag} and clearing load_images",
                              className=Upload.Formatting(color='success')))
          Upload.initDir(f"{Upload.image_dir}/{tag}")

          for key in fileDict.keys():
            print(f"Writing {key} to {Upload.image_dir}/{tag}/")
            imwrite(
                f"{Upload.image_dir}/{tag}/{key}", 
                fileDict[key]["img"])
            
            if hasMask: 
              k = key.replace('.', '_mask.')
              print(f"Writing {k} to {Upload.image_dir}/{tag}/")
              imwrite(
                  f"{Upload.image_dir}/{tag}/{k}", 
                  fileDict[key]["mask"])

          print(f"Clearing {Upload.load_dir}")
          for item in os.listdir(Upload.load_dir):
            path = os.path.join(Upload.load_dir, item)
            if os.path.isfile(path): os.remove(path)
            else: shutil.rmtree(path)
        
        if val and len(fileDict.keys())*2 == fullVal and hasMask and len(fileDict.keys())>0: WriteWrap()
        elif val and len(fileDict.keys()) == fullVal and not hasMask and len(fileDict.keys())>0: WriteWrap()
        elif not val and len(fileDict.keys()) > 1: WriteWrap()
        else: mdiv.append(html.H2(f"not writing to {tag}, not clearing load_images",
                                  className=Upload.Formatting(color='success')))

        if hasMask: mdiv.append(html.H2("Accepted image/mask pairs:"))
        else: mdiv.append(html.H2("Accepted images:"))

        for key in fileDict.keys():
          ro = fileDict[key]
          rS = (450, 450)
          rImg = resize(ro["img"], rS)
          if hasMask:
            mdiv.extend([
                html.H5(key),
                dbc.Row([
                    dbc.Col([
                        Upload.PlotImage(rImg, h=600, w=650)
                    ], width=6),
                    dbc.Col([
                        Upload.TI2(rImg, resize(ro["mask"], rS), h=600, w=650)
                    ], width=6), 
                ], align='justify'),
            ])
          else: mdiv.extend([html.H5(key), Upload.PlotImage(rImg, h=600, w=800)])

        if hasMask: mdiv.append(html.H2("Rejcted image/mask pairs:"))
        else: mdiv.append(html.H2("Rejected images:"))

        if len(reject) == 0: mdiv.append(html.H5("No rejected images!"))
        else: 
          for r in reject: 
            mdiv.append(html.H2(r))

      else: 
        rules = dcc.Markdown(f'''
          * Image(s)/mask(s) must be: 2-dimensional *

          1. Set tag
              ```
              - The name of the sub-directory in image_data where the images/masks will be stored
              - If writing to existing directory, it will write over any files with identical name
              
              - Tags in use: 
                {os.listdir('./vol/image_data')}
              ```

          2. Has Mask
              ```
              if True:
                - Requires each image to have a mask and the mask name is the image name with _mask at the end
                
                - Example: image named 001.tif would have a mask named 001_mask.tif (or .png)
              ```

          4. Complete Validation
              ```
              If True: 
                requires all files to load to be accepted before
                writing to image_data and clearing image_loader
              ```
          5. Non-segmented imageJ masks:
            - If True:
                - Creates segmentation mask using [scipy.ndimage.label](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.label.html)
                - **Requires all mask files to be non-segmented imageJ**
            - Else:
                - Assumes mask files are generated by this gui or the Cellpose gui

          ''', 
        style={
            'backgroundColor': '#121212',
            'color': '#FFFFFF',       
            'padding': '20px',     
          }
        )

        examples = html.Div([
          html.H3("Supported Upload Formats",
                  className=Upload.Formatting()),
          
          dbc.Row([
            dbc.Col([
                Upload.PI3(Upload.ReadImage("nonSeg.png", 'pages/assets'), 500, 500, False),
                dcc.Markdown('''
                    - Mask generated by ImageJ
                    - Requires: "Has Mask" == True and "Non-segmented imageJ masks" == True
                  ''')
            ]),
            dbc.Col([
                Upload.PI3(Upload.ReadImage("tif_png.png", 'pages/assets'), 700, 700, False),
                dcc.Markdown('''
                    - Mask files generated by this app. Mask files containing _cp_ were mask files 
                      that were corrected in the cellpose GUI
                      
                    - Requires: "Has Mask" == True and "Non-segmented imageJ masks" == False
                  '''),
            ]), 
          ], align='justify'),

          dbc.Row([
              Upload.PI3(Upload.ReadImage("vsi.png", 'pages/assets'), 1000, 1000, False),
              dcc.Markdown('''
                  <ul style="list-style-position: inside; text-align: center;">
                      <li>Accepts raw microspy data with x.vsi and _x/stack1/frame_t_0.ets file format</li>
                      <li>Requires: "Has Mask" == False</li>
                  </ul>
              ''', dangerously_allow_html=True)
          ], align='center'),

          html.H3("Correct Uploads",
                  className=Upload.Formatting()),

          dbc.Row([
            dbc.Col([
                Upload.PI3(Upload.ReadImage("p1.png", 'pages/assets'), 700, 700, False)
            ]),
            dbc.Col([
                Upload.PI3(Upload.ReadImage("p2.png", 'pages/assets'), 700, 700, False)
            ]), 
            dbc.Col([
                Upload.PI3(Upload.ReadImage("p3.png", 'pages/assets'), 700, 700, False)
            ]), 
          ], align='justify'),
          
          html.H3("Incorrect Uploads",
                  className=Upload.Formatting()),

          dbc.Row([
            dbc.Col([
                Upload.PI3(Upload.ReadImage("b1.png", 'pages/assets'), 700, 700, False)
            ]),
            dbc.Col([
                Upload.PI3(Upload.ReadImage("b2.png", 'pages/assets'), 700, 700, False)
            ]),
          ], align='justify'),
        ])

        mdiv.extend([rules, examples])  
      
      print("plotting...")
      return (mdiv, tag, hasMask, val, imageJ)
          
x = Upload()
layout = x.layout()
x.callbacks()
