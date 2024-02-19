import plotly.graph_objects as go
import numpy as np
import os 

from tifffile import imread, imwrite
from skimage.io import imread as pngRead
from skimage.segmentation import mark_boundaries
from scipy.ndimage import label 
from dash import dcc, dash_table
from PIL import Image

import io
import base64

class DashUtil:

  @staticmethod
  def DarkDashTable(df, rows: int = 30):
    return dash_table.DataTable(
            data = df.to_dict('records'),
            columns = [{'name': i, 'id': i} for i in df.columns],
            export_format="csv",
            sort_action='native', 
            page_size=rows,
            filter_action='native',
            style_header={'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white'},
            style_data={'backgroundColor': 'rgb(50, 50, 50)','color': 'white'},
            style_table={'overflowX': 'auto'},
            style_cell={
                'height': 'auto',
                'minWidth': '70px', 'width': '70px', 'maxWidth': '180px',
                'whiteSpace': 'normal'
              }
          ) 

  @staticmethod
  def PlotImage(img: np.ndarray, colorscale: str = 'greys') -> dcc.Graph:
    return dcc.Graph(figure=go.Figure(
                        go.Heatmap(
                          z=img, colorscale=colorscale
                        )).update_layout(
                              margin=dict(l=0.1, r=0.1, t=0.1, b=0.1), 
                              height=450, width=450
                        ))

  @staticmethod
  def PI2(img: np.ndarray) -> dcc.Graph:
    buffer = io.BytesIO()
    Image.fromarray(img
        ).transpose(Image.FLIP_TOP_BOTTOM
        ).save(buffer, format='PNG')

    encoded_img = base64.b64encode(buffer.getvalue()).decode('utf-8')

    figure = go.Figure(go.Image(source=f"data:image/png;base64,{encoded_img}"))

    figure.update_layout(
        margin=dict(l=0.1, r=0.1, t=0.1, b=0.1), 
        height=450, width=450
    )

    return dcc.Graph(figure=figure)

  @staticmethod
  def TransparentImage(img, mask, 
                      colorscale: str = 'emrld',
                      colorscale_interp: bool = False
                    ):
    
    img, transparent_mask = img, mask.astype(float) 
    transparent_mask[transparent_mask == 0] = np.nan

    if colorscale_interp: zmin, zmax = 0, np.max(mask)+1
    else: zmin, zmax = None, None

    return dcc.Graph(figure=go.Figure(
                      go.Heatmap(z=np.where(np.isnan(transparent_mask), 
                                img, 
                                transparent_mask), 
                      zmin=zmin, 
                      zmax=zmax,
                      colorscale=colorscale
                    )).update_layout(
                        margin=dict(l=0.1, r=0.1, t=0.1, b=0.1), 
                        height=450, width=650)
                  )

  @staticmethod
  def TI2(img, mask):    
    buffer = io.BytesIO()
    Image.fromarray(
            (mark_boundaries(img, mask)*255).astype(np.uint8)
          ).transpose(Image.FLIP_TOP_BOTTOM
          ).save(buffer, format="PNG")
    
    encImage = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    figure = go.Figure()
    figure.add_trace(go.Image(source=f"data:image/png;base64,{encImage}"))
    
    figure.update_layout(
        margin=dict(l=0.1, r=0.1, t=0.1, b=0.1), 
        height=450, width=450
    )

    return dcc.Graph(figure=figure)

  @staticmethod
  def Formatting( 
                className: str = 'heading', 
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

class Preprocessing:

  @staticmethod
  def ReadImage(name, directory):
    if ".tif" in name: return imread(f"{directory}/{name}") 
    if ".png" in name: return pngRead(f"{directory}/{name}") 

  @staticmethod
  def SegmentMask(mask):
    mask[(mask > 1) & (mask <= 255)] = 1
    mask, features = label(mask) 
    return mask
  
  @staticmethod
  def DesegmentMask(mask):
    return np.where(mask > 0, 255, mask)

  @staticmethod
  def PixelAccuracy(mask, pmask):
      return np.sum(mask == pmask) / mask.size

  @staticmethod
  def DiceCoefficient(mask, pmask):
      mask = mask.flatten()
      pmask = pmask.flatten()
      
      intersection = np.sum(mask * pmask)
      union = np.sum(mask) + np.sum(pmask)
      
      return (2.0 * intersection) / (union + 1e-10)  

  @staticmethod
  def initDir(dir_path):
    if not os.path.exists(dir_path): os.makedirs(dir_path)
    else: print(f"Directory: {dir_path} exists")

  @staticmethod
  def NameCleaner(file: str):
    file = file.lower(
            ).replace(" ", ""
            ).replace("_", ""         
            ).replace("groundtruth", "tmp"
            ).replace("mask", "tmp")
    
    ## Make sure _mask is at end of file
    if "tmp" in file: 
      file = file.replace(".", "_mask.").replace("tmp", "")
    return file

  @staticmethod
  def TorchGPU():
    import torch

    return ", ".join([
        f"CUDA GPU enabled: {torch.cuda.is_available()}",
        f"CUDA version: {torch.version.cuda}"
      ])

if __name__ == "__main__":
  gpuEnabled = True

  x = Preprocessing()
  if gpuEnabled: x.TorchGPU()

