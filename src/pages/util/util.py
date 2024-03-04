import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os 
import datetime

from tifffile import imread, imwrite
from skimage.io import imread as pngRead
from skimage.segmentation import mark_boundaries
from scipy.ndimage import label 
from dash import dcc, dash_table, html
from PIL import Image
from aicsimageio import AICSImage

import io
import base64
import os

class DashUtil:

  @staticmethod
  def EvalAggTransforms(df: pd.DataFrame):
    df = df.groupby(by=["model", "diam mean"], as_index=True
                        ).agg({"jaccard index": ["count", 'mean', 'std'],
                                "euclidean normalized rmse": ['mean'],
                                "structural similarity": ['mean'],
                                "average precision": ["mean"],
                                "true positives": ["mean"],
                                "false positives": ["mean"],
                                "false negatives": ["mean"]
                        }).reset_index(
                        ).round(5)  
    df.columns = df.columns.map(' '.join)
    return df.sort_values("euclidean normalized rmse mean",
          ).rename(columns={"jaccard index count": 'sample size'})


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
  def PlotImage(img: np.ndarray, colorscale: str = 'greys',
                h:int=450, w:int=450) -> dcc.Graph:
    return dcc.Graph(figure=go.Figure(
                        go.Heatmap(
                          z=img, colorscale=colorscale
                        )).update_layout(
                              margin=dict(l=0.1, r=0.1, t=0.1, b=0.1), 
                              height=h, width=w
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
  def PI3(img: np.ndarray, h: int = 450, w: int = 450, flip: bool = True) -> html.Div:
      buffer = io.BytesIO()
      
      if flip: Image.fromarray(img).transpose(Image.FLIP_TOP_BOTTOM).save(buffer, format='PNG')
      else: Image.fromarray(img).save(buffer, format='PNG')

      encoded_img = base64.b64encode(buffer.getvalue()).decode('utf-8')

      return html.Div(
              html.Img(
                src=f"data:image/png;base64,{encoded_img}",
                style={'max-height': f'{h}px', 'max-width': f'{w}px', 
                      'width': 'auto', 'height': 'auto', 
                      'display': 'block', 'margin': 'auto'}
                )
              )

  @staticmethod
  def TransparentImage(img, mask, 
                      colorscale: str = 'emrld',
                      colorscale_interp: bool = False,
                      h:int=450, w:int=650
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
                        height=h, width=w)
                  )

  @staticmethod
  def TI2(img, mask, h:int=450, w:int=450):    
    buffer = io.BytesIO()
    Image.fromarray(
            (mark_boundaries(img, mask, mode='thick')*255).astype(np.uint8)
          ).transpose(Image.FLIP_TOP_BOTTOM
          ).save(buffer, format="PNG")
    
    encImage = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    figure = go.Figure()
    figure.add_trace(go.Image(source=f"data:image/png;base64,{encImage}"))
    
    figure.update_layout(
        margin=dict(l=0.1, r=0.1, t=0.1, b=0.1), 
        height=h, width=w
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
  def StrTime(asInt: bool = False):
    x = int(datetime.datetime.utcnow().timestamp())
    if asInt: return x
    else: return str(x)

  @staticmethod
  def VsiToTif(readDir: str = '.', writeDir = '.'):
    for filename in [f for f in os.listdir(readDir) if f.endswith('.vsi')]:
      img = AICSImage(f"{readDir}/{filename}")
      for channel in range(img.shape[0]):
        fname = filename.split('/')[-1].split('.')[0]
        name = f'{writeDir}/{fname}_ch{channel}.tif'
        imwrite(
          name, 
          img.get_image_data("CZYX", C=channel)[0, 0, :, :]
        )
        print(f"Wrote {name.split('/')[-1]} to {writeDir}")
        
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
            ).replace("_cp_masks", "tmp"
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

  @staticmethod
  def ArrayCheck(z: np.ndarray, x: str):
    if z.dtype == np.int32:
      if np.any(z > np.iinfo(np.uint16).max):
        print(f"Dataloss may occur for {x} conversion: int32 -> uint16")
      else: 
        print(f"Coverted {x} from uint32 to uint16")
      return z.astype(np.uint16)
    else: return z


if __name__ == "__main__":
  gpuEnabled = True

  x = Preprocessing()
  if gpuEnabled: x.TorchGPU()

