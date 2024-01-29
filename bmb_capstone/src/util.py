from tifffile import imread, imwrite

#from cellpose.models import Cellpose, CellposeModel
#from cellpose.plot import mask_overlay
#import cellpose
import os 

class Preprocessing:

  def __init__(self, export_dir: str = './data/training') -> None:
    self.export_dir = export_dir

  def CleanFileNames(self,
                import_dir: str = './imageLoad', # import directory abs path
              ):
    
    eFiles = os.listdir(self.export_dir)
    if len(eFiles) >= 1: 
      [os.remove(f"{self.export_dir}/{x}") for x in eFiles]
      print(f"deleted {len(eFiles)} files from ./data")

    iFiles, filesAdded = os.listdir(import_dir), 0
    for file in iFiles:
      if ".tif" in file:
        filesAdded += 1
        name = file.lower(
            ).replace(" ", "_"
            ).replace("groundtruth", "remove"
            ).replace("dict", ""
            ).replace("dic", "")
            
        if "remove" in name: name = name.replace(".", "_mask."
                                        ).replace("remove", "")
            
        imwrite(
          f'{self.export_dir}/{name}',
          imread(f"{import_dir}/{file}")  
        )
      
    print(f"saved {filesAdded} files to ./data")
  
  def VerifyFiles(self, debug: bool = False):
    if not debug: Preprocessing.CleanFileNames(self)

    fileDict = {x:None for x in os.listdir(self.export_dir) if "seg" not in x}
    for x in fileDict.keys():
      try:
        if "mask" in x: fileDict[x.replace("_mask", "")]
        else: fileDict[x.replace(".", "_mask.")]
      except: raise Exception(f"file: {x} is missing counterpart file")

  def ImageShape(self):
    for file in os.listdir(self.export_dir):
      img = imread(f"{self.export_dir}/{file}")

      print(f"Array:", img, 
            f"Shape:{img.shape}", 
            F"Channels: {img.ndim}", 
          sep='\n')
      break

  @staticmethod
  def TorchGPU():
    import torch

    print("CUDA availability:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)

if __name__ == "__main__":
  loadFiles = False

  x = Preprocessing()
  if loadFiles: x.VerifyFiles()
  x.ImageShape()
  x.TorchGPU()