from tifffile import imread, imwrite
from skimage.io import imread as pngRead
from cellpose import models
from cellpose.metrics import aggregated_jaccard_index
from os.path import abspath, exists
from os import listdir, makedirs
from time import time
from datetime import datetime
from cv2 import resize
from typing import Optional
import numpy as np

class Schmoo:
    
  def __init__(self, 
              useGpu: bool = True,
              model_dir: str = './models',
              data_dir: str = './data/tania',
              predict_dir: str = './predictions',
              diam_mean: float = 30,
              ) -> None:
    
    self.timeStart = time()
    self.gpu = useGpu        
    self.model_dir = model_dir
    self.data_dir = data_dir
    self.predict_dir = predict_dir
    self.diam_mean = diam_mean
      
  def initDir(self, dir_path):
    if not exists(dir_path): makedirs(dir_path)
    else: print(f"Directory: {dir_path} exists")
  
  def DataGenerator(self, maskRequired: bool = True):
    self.dataGen = {}
    print('=== init data generator ===')
    
    for x in listdir(self.data_dir):
      if maskRequired:
        if not "mask" in x:
          img = imread(f"{self.data_dir}/{x}")
          
          try:  mask = imread(f"{self.data_dir}/{x.replace('.', '_mask.')}")
          except: 
            y = x.replace('.', '_mask.').replace('.tif', '.png')
            
            try: mask = pngRead(f"{self.data_dir}/{y}")
            except: raise Exception(f"Mask file not found for {x}")

          self.dataGen[x] = {'img':img, 'mask':mask, 'pmask': None}
      else:
        if not "mask" in x:
          img = imread(f"{self.data_dir}/{x}")
          self.dataGen[x] = {'img':img, 'mask':None, 'pmask': None}
    
    print('=== returned data generator ===')
    return self.dataGen

  def TrainModel(self,
                model_type: str = 'cyto',
                image_channels: list[int] = [0,0], # gray scale images
                learning_rate: float = .2,
                weight_decay: float = .00001,
                n_epochs: int  = 250, 
                save_every: int = 50, # save after amount of epochs
                min_train_masks: int = 5,
                residual_on: bool = True, # celloose Unet if True
                rescale: bool = True,
                normalize: bool = True,
                savePath: str = './'
            ):

    Schmoo.DataGenerator(self)
    train_data, train_labels = [], []
    for key in self.dataGen.keys():
      train_data.append(self.dataGen[key]["img"])
      train_labels.append(self.dataGen[key]["mask"])

    name = "_".join([model_type,
                    f"lr{int(learning_rate*100)}",
                    f"wd{int(learning_rate*100000)}",
                    f"ep{n_epochs}", 
                    str(time()).split('.')[1]
                ])

    print(f'=== init training model: {name} ===')
    models.CellposeModel(
                        gpu=self.gpu,
                        model_type=model_type,
                        diam_mean=self.diam_mean
                      ).train(train_data=train_data, 
                              train_labels=train_labels,
                              channels=image_channels, 
                              learning_rate=learning_rate, 
                              weight_decay=weight_decay, 
                              n_epochs=n_epochs,
                              normalize=normalize,
                              rescale=rescale,
                              save_every=save_every,
                              min_train_masks=min_train_masks,
                              save_path=savePath,
                              model_name=name
                            )
    
    print(f"=== saved {name} to ./ after {time()-self.timeStart} seconds ===")

  def TestModel(self,
              model_name: str = 'cyto2torch_0',
              image_channels: list[int] = [0,0],
              numPredictions: Optional[int] = None,
              imgResize: tuple = (450, 450),
              saveImages: bool = True,
              figures: bool = True,
              getData: bool = True,
          ):
    
    if getData: Schmoo.DataGenerator(self, maskRequired=False)

    if model_name in listdir(self.model_dir):
            
        model = models.CellposeModel(pretrained_model=f"{self.model_dir}/{model_name}",
                                    gpu=self.gpu,
                                )
        print(f'Opened model: {model_name}')

        stringTime = datetime.now().strftime('%Y-%m-%d_%H').replace('-', '_')
        savePath = f"{self.predict_dir}/{stringTime}_{model_name}"    

        if (len(self.dataGen.keys()) > 0) and saveImages: 
          Schmoo.initDir(self, savePath)
          print(f"Saving images to: {savePath}")
            
        fig, stat, count = [], [], 0 
        for key in self.dataGen.keys():
          filename, img = key, self.dataGen[key]["img"]
          pmask, flow, styles = model.eval(img,
                                          channels=image_channels,
                                          rescale=True,
                                          diameter=self.diam_mean
                                        )
          
          print(f"Generated mask for {filename}")
          self.dataGen[key]["pmask"] = pmask
          
          if round(pmask.mean(),5) > 0:
            if saveImages:
              imwrite(f"{savePath}/img_{filename}", img)
              imwrite(f"{savePath}/mask_{filename}", pmask)
              print(f"Saved img/mask for {filename}")
            
            if figures: 
              rImg, rMask = resize(img, imgResize), resize(pmask, imgResize)
              fig.append([x for x in [rImg, rMask, filename]])

          else: print(f"No mask found for {filename}")

          count += 1
          if (numPredictions != None) and (count >= numPredictions): break 
    else: raise Exception(f"{model_name}: not found in {self.model_dir}")
    if len(fig) > 0: return fig
  
  def ModelStats(self, 
                modelName: str = 'cyto_lr20_wd20000_ep100_7559516',
                numPredictions: int = 2,
              ):
    
    Schmoo.DataGenerator(self)
    Schmoo.TestModel(self,
                    model_name=modelName, 
                    numPredictions=numPredictions,
                    getData=False,
                  )
    
    masks, pmasks = [], []
    for key in self.dataGen.keys():
      pmask = self.dataGen[key]["pmask"]
      if isinstance(pmask, np.ndarray):
        masks.append(self.dataGen[key]["mask"])
        pmasks.append(pmask)

    return [modelName, aggregated_jaccard_index(masks, pmasks)]

if __name__ == "__main__":
  x = Schmoo(model_dir='../../models',
            data_dir='../../data/tania', 
            predict_dir='../../predictions', 
            diam_mean=80
          )
  
  dataGen, train, test, stats = False, True, False, False

  if dataGen:
      data = x.DataGenerator()
      print(len(data.keys()))
      for key in data.keys():
        image, mask = data[key]["img"], data[key]["mask"] 
        print(image, mask, key, sep='\n')
        break
  
  if train: x.TrainModel(savePath='../..')
  if test: x.TestModel(model_name='cyto_lr20_wd20000_ep100_7559516', 
                      numPredictions=5)
    
  if stats: x.ModelStats()