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
    Eclipse.DataGen(self, skip=4)

    frames, exposures = [], []
    for k, v in self.imgs.items():
      frames.append(v["frame"])
      exposures.append(v["exposure"])

      imwrite(f"tif_data/{k}.tif", v["frame"])

    if hasattr(self, 'imgs'):
      if self.imgs:
        alignMTB = cv2.createAlignMTB()
        alignMTB.process(frames, frames)

        # Create HDR image
        mergeDebevec = cv2.createMergeDebevec()
        hdr = mergeDebevec.process(
                            frames, 
                            cv2.UMat(np.array(exposures, dtype=np.float32))
                          )

        cv2.imwrite('export.hdr', hdr)

  def Plot_HDR(self):
    hdr_image = cv2.imread('exp.hdr', cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)

    if hdr_image is not None:
      plt.imshow(cv2.cvtColor(hdr_image, cv2.COLOR_BGR2RGB))
      plt.title('HDR Image')
      plt.axis('off')
      plt.show()

if __name__ == "__main__":
  x = Eclipse()
  x.Write_HDR()


