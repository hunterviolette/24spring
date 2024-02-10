import plotly.graph_objects as go
import numpy as np
import os 

from tifffile import imread, imwrite
from skimage.io import imread as pngRead
from scipy.ndimage import label 

class DashUtil:

  @staticmethod
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

  def __init__(self, 
              import_dir: str = './raw_data/tania',
              export_dir: str = './data/tania',) -> None:
    self.import_dir = import_dir
    self.export_dir = export_dir

  def Cleaner(self, file: str):
    dir = self.import_dir.split('/')[-1]

    if dir in ['original', 'ws_set2']:
      return file.lower(
                ).replace(" ", "_"
                ).replace("groundtruth", "remove"
                ).replace("dict", ""
                ).replace("dic", "")

    elif dir in ['tania', 'tania_unlabeled']:
      return file.lower(
          ).replace("_ dict", "_dict"
          ).replace(" ", "_"
          ).replace("masks", "mask"
          ).replace("_dict", "")
    
  def CleanFileNames(self,):
    
    eFiles = os.listdir(self.export_dir)
    if len(eFiles) >= 1: 
      [os.remove(f"{self.export_dir}/{x}") for x in eFiles]
      print(f"deleted {len(eFiles)} files from {self.export_dir}")

    iFiles, filesAdded = os.listdir(self.import_dir), 0
    for file in iFiles:
      filesAdded += 1
      name = Preprocessing.Cleaner(self, file)
      
      if ".tif" in file:  
        img = imread(f"{self.import_dir}/{file}") 

        if "remove" in name: 
          name = name.replace(".", "_mask.").replace("remove", "")
          img[(img > 1) & (img <= 255)] = 1
          img, features = label(img) 
      
      if ".png" in file:
        img = pngRead(f"{self.import_dir}/{file}") 

      imwrite(f'{self.export_dir}/{name}', img)
      
    print(f"saved {filesAdded} files to {self.export_dir}")
  
  def VerifyFiles(self,
                  debug: bool = False,
                ):
    if not debug: Preprocessing.CleanFileNames(self)

    fileDict = {x:None for x in os.listdir(self.export_dir) if "seg" not in x}
    for x in fileDict.keys():
      try:
        if "mask" in x: fileDict[x.replace("_mask", "")]
        else: fileDict[x.replace(".", "_mask.")]
      except: 
        try:
          if "mask" in x: fileDict[x.replace("_mask", "").replace('.png', '.tif')]
          else: fileDict[x.replace(".", "_mask.").replace('.tif', '.png')]
        except: raise Exception(f"file {x} is missing counterpart file")
    
    print(f"All files in {self.export_dir} validated")

  @staticmethod
  def TorchGPU():
    import torch

    print("CUDA availability:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)

if __name__ == "__main__":
  loadFiles, gpuEnabled = True, False

  x = Preprocessing(import_dir='../../raw_data/tania_unlabeled', 
                    export_dir='../../data/tania_unlabeled')
  
  if loadFiles: x.VerifyFiles()
  if gpuEnabled: x.TorchGPU()

