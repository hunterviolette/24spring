import cv2
import numpy as np
import os
import rawpy
from tifffile import imwrite, imread
import pyexiv2
import matplotlib.pyplot as plt
import itertools

class Eclipse:
  
  def __init__(self) -> None:
    pass

  def DataGen(self, input_dir: str = './cr2_cut', skip: int = 2):
    
    self.imgs = {}
    for i, file in enumerate([
        x for x in os.listdir(input_dir) 
        if x.lower().endswith('.cr2')
        ]):
      
      if i % skip == 0:

        file_path = os.path.join(input_dir, file)
        metadata = pyexiv2.Image(file_path).read_exif()

        print(f"loaded {file}")                  
        self.imgs[file.split(".")[0]] = {
          'frame': rawpy.imread(file_path).postprocess(),
          "exposure": eval(metadata['Exif.Photo.ExposureTime'])
        }

  def Write_HDR(self, 
                  name: str , 
                  type: str = "debevec", 
                  gamma_correction: float = 1,
                  tonemapping: bool = False,
                  export_dir: str = './export'
                ):
  
    if not hasattr(self, 'imgs'):
      self.DataGen(skip=1)

    images, pre_exposures = [], []
    for k, v in self.imgs.items():
      images.append(v["frame"])
      pre_exposures.append(v["exposure"])

      imwrite(f"tif_data/{k}.tif", v["frame"])

    if hasattr(self, 'imgs'):
      if self.imgs:
        exposures = np.array(pre_exposures, dtype=np.float32)

        if type.lower() == "mertens":
          merge_mertens = cv2.createMergeMertens()
          hdr = merge_mertens.process(images)
        elif type.lower() == "robertson":
          merge_robertson = cv2.createMergeRobertson()
          hdr = merge_robertson.process(images, times=exposures.copy())
        elif type.lower() == "debevec":
          calibrate = cv2.createCalibrateDebevec()
          response = calibrate.process(images, exposures)

          merge_debevec = cv2.createMergeDebevec()
          hdr = merge_debevec.process(images, exposures.copy(), response)
        else:
          raise TypeError("Invalid HDR type")

        if tonemapping:
          tonemap = cv2.createTonemap(gamma=2.2)
          hdr = tonemap.process(hdr.copy())

        if gamma_correction != 1:
          # The functionality would be the same without this if statement, but not the performance.
          hdr = (hdr/hdr.max()) ** gamma_correction * hdr.max()

        cv2.imwrite(f'{export_dir}/{name}.hdr', hdr)

if __name__ == "__main__":
  x = Eclipse()
  x.DataGen('./cr2_cut', skip=1)
  
  for hdr_type, tm, g in itertools.product(
                                    ['mertens'],#, 'robertson', 'debevec'], 
                                    [False],#, True], 
                                    np.linspace(3, 10, 8)
                                  ):
    
    x.Write_HDR(
      name="_".join([hdr_type, str(tm), str(g)]),
      type=hdr_type,
      gamma_correction=g,
      tonemapping=tm
    )