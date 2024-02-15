import plotly.graph_objects as go
import numpy as np
import os 

from tifffile import imread, imwrite
from skimage.io import imread as pngRead
from scipy.ndimage import label 
from dash import dcc

class DashUtil:

  @staticmethod
  def PlotImage(img: np.ndarray, colorscale: str = 'greys') -> dcc.Graph:
    return dcc.Graph(figure=go.Figure(
                        go.Heatmap(
                          z=img, colorscale=colorscale
                        )).update_layout(
                              margin=dict(l=0.1, r=0.1, t=0.1, b=0.1), 
                              height=450, width=650
                        ))

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

