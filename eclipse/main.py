import cv2
import numpy as np
import os
import rawpy
from tifffile import imwrite, imread
import pyexiv2
import matplotlib.pyplot as plt

class Eclipse:
  
  def __init__(self) -> None:
    pass

  def DataGen(self, input_dir: str = './cr2_data', skip: int = 2):
    self.imgs = {}

    files = [x for x in os.listdir(input_dir) if x.lower().endswith('.cr2')]
    for i, file_name in enumerate(files):
      if i % skip == 0:
        file_path = os.path.join(input_dir, file_name)
        
        metadata = pyexiv2.Image(file_path).read_exif()
                          
        name = file_name.split(".")[0]
        self.imgs[name] = {
            'frame': rawpy.imread(file_path).postprocess(),
            "exposure": eval(metadata['Exif.Photo.ExposureTime'])
        }

  def Write_HDR(self):
    self.DataGen(skip=3)

    frames, pre_exposures = [], []
    for k, v in self.imgs.items():
      frames.append(v["frame"])
      pre_exposures.append(v["exposure"])

      imwrite(f"tif_data/{k}.tif", v["frame"])

    if hasattr(self, 'imgs') and self.imgs:
      exposures = np.array(pre_exposures, dtype=np.float32)

      calibrate = cv2.createCalibrateDebevec()
      response = calibrate.process(frames, exposures)
      merge_debevec = cv2.createMergeDebevec()
    
      hdr = merge_debevec.process(frames, exposures, response)
      tonemap = cv2.createTonemap(2.2)
      
      ldr = tonemap.process(hdr)
      ldr_np = np.clip(ldr * 255, 0, 255).astype(np.uint8)

      merge_mertens = cv2.createMergeMertens()
      fusion = merge_mertens.process(frames)

      cv2.imwrite('fusion.png', fusion * 255)
      cv2.imwrite('ldr.png', ldr_np)
      cv2.imwrite('hdr.hdr', hdr)

if __name__ == "__main__":
  x = Eclipse()
  x.Write_HDR()
