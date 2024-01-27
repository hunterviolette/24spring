from tifffile import imread, imwrite

#from cellpose.models import Cellpose, CellposeModel
#from cellpose.plot import mask_overlay
#import cellpose
import os 

class Preprocessing:

  def __init__(self, export_dir: str = './data') -> None:
    self.export_dir = export_dir

  def CleanFileNames(self,
                import_dir: str = './imageLoad', # import directory abs path
              ):
    
    eFiles = os.listdir(self.export_dir)
    if len(eFiles) >= 1: 
      [os.remove(f"{self.export_dir}/{x}") for x in eFiles]
      print(f"deleted {len(eFiles)} files from ./data")

    iFiles = os.listdir(import_dir)
    for file in iFiles:
      name = file.lower(
          ).replace(" ", "_"
          ).replace("groundtruth", "mask_"
          ).replace("dict", ""
          ).replace("dic", "")
      imwrite(
        f'{self.export_dir}/{name}',
        imread(f"{import_dir}/{file}")  
      )
    
    print(f"saved {len(iFiles)} files to ./data")
  
  def VerifyFiles(self, debug: bool = False):
    if not debug: Preprocessing.CleanFileNames(self)

    fileDict = {x:None for x in os.listdir(self.export_dir) if "seg" not in x}
    for x in fileDict.keys():
      try:
        if "mask" in x: fileDict[x.replace("mask_", "")]
        else:
          nam = x.split("_")
          nam[-1] = f"mask_{nam[-1]}"
          fileDict["_".join(nam)]
      except: raise Exception(f"file: {x} is missing counterpart file")

  def ImageShape(self):
    for file in os.listdir(self.export_dir):
      img = imread(f"{self.export_dir}/{file}")
      
      print(f"Array:", img, 
            f"Shape:{img.shape}", 
            F"Channels: {img.ndim}", 
          sep='\n')
      break

if __name__ == "__main__":
  x = Preprocessing()
  x.VerifyFiles(True)
  x.ImageShape()