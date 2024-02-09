import numpy as np
import pandas as pd
import os

from time import time
from datetime import datetime
from typing import Optional

from tifffile import imread, imwrite
from skimage.io import imread as pngRead
from cv2 import resize

from cellpose import models
from cellpose.metrics import aggregated_jaccard_index

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
    if not os.path.exists(dir_path): os.makedirs(dir_path)
    else: print(f"Directory: {dir_path} exists")
  
  def DataGenerator(self, maskRequired: bool = True):
    self.dataGen = {}
    print('=== init data generator ===')
    
    for x in os.listdir(self.data_dir):
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

  def DataGenList(self, 
                  listType: str # Takes img, mask, pmask
                ):
    if not hasattr(self, 'dataGen'): Schmoo.DataGenerator(self)    

    if listType in ["img", "mask", "pmask"]:
      return list(map(lambda key: self.dataGen[key][listType], self.dataGen.keys()))
    else: raise Exception("listType only accepts img, mask, pmask")

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

    if not hasattr(self, 'dataGen'): Schmoo.DataGenerator(self)

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
                      ).train(train_data=Schmoo.DataGenList(self, 'img'), 
                              train_labels=Schmoo.DataGenList(self, 'mask'),
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

  def ModelPredict(self,
              model_name: str = 'cyto2torch_0',
              image_channels: list[int] = [0,0], # Assumes 2 channel greyscale
              numPredictions: Optional[int] = None, # Number of predictions before breaking loop
              imgResize: Optional[int] = None, # int value in pixels  
              saveImages: bool = False,
              figures: bool = True, # Return list of images 
              hasTruth: bool = False, # Has true masks for images 
              modelPath: Optional[str] = None
            ):
    
    if not hasattr(self, 'dataGen'): Schmoo.DataGenerator(self)

    if modelPath == None: dir = self.model_dir
    else: dir = modelPath

    if model_name in os.listdir(dir):
            
      model = models.CellposeModel(pretrained_model=f"{dir}/{model_name}",
                                  gpu=self.gpu,
                              )
      print(f'Opened model: {model_name}')

      stringTime = datetime.now().strftime('%Y-%m-%d_%H').replace('-', '_')
      savePath = f"{self.predict_dir}/{stringTime}_{model_name}"    

      if (len(self.dataGen.keys()) > 0) and saveImages: 
        Schmoo.initDir(self, savePath)
        print(f"Saving images to: {savePath}")
          
      fig, count = [], 0 
      for key in self.dataGen.keys():
        filename, img = key, self.dataGen[key]["img"]
        pmask, flow, styles = model.eval(img,
                                        channels=image_channels,
                                        rescale=True,
                                        diameter=self.diam_mean
                                      )
        
        print(f"Generated mask for {filename}")
        self.dataGen[key]["pmask"] = pmask
                
        if saveImages:
          imwrite(f"{savePath}/img_{filename}", img)
          imwrite(f"{savePath}/mask_{filename}", pmask)
          print(f"Saved img/mask for {filename}")
        
        if figures: 
          if isinstance(imgResize, int):
            if imgResize == 0: pass
            else:
              tupleSize = (imgResize, imgResize)
              img, pmask = resize(img, tupleSize), resize(pmask, tupleSize)

          if hasTruth: fig.append([img, pmask, filename, 
                                  Schmoo.ModelEval])
            
          else: fig.append([img, pmask, filename])

        count += 1
        if (numPredictions != None) and (count >= numPredictions): break 

      if len(fig) > 0: return fig

    else: raise Exception(f"{model_name}: not found in {dir}")
  
  def ModelEval(self, 
                predMask: Optional[np.ndarray] = None,
                trueMask: Optional[np.ndarray] = None,
                numPredictions: Optional[int] = None,
                modelName: Optional[str] = 'cyto_lr20_wd20000_ep100_7559516',
                singleEval: bool = False,
                modelPath: Optional[str] = None,
              ):
    
    if not hasattr(self, 'dataGen'): Schmoo.DataGenerator(self)

    if not singleEval:
      Schmoo.ModelPredict(self, 
                          model_name=modelName,
                          numPredictions=numPredictions,
                          modelPath=modelPath
                        )
      
      self.dataGen = {key: value for key, value in self.dataGen.items() \
                      if value['pmask'] is not None}

      return [modelName,
              aggregated_jaccard_index(
                  Schmoo.DataGenList(self, 'mask'),
                  Schmoo.DataGenList(self, 'pmask') 
              )]
    else: 
      if trueMask == None or predMask == None:
        raise Exception("Single Eval True requires predMask/trueMask")
      return aggregated_jaccard_index(trueMask, predMask)


  def BatchTrainModel(self):
    pass

  def CompareModels(self, 
                    modelDir: str = './saved_models',
                    numPredictions: Optional[int] = None,
                  ):
    
    Schmoo.DataGenerator(self)
    
    df = pd.DataFrame()
    print(os.listdir(modelDir))
    for name in os.listdir(modelDir):
      ret = Schmoo.ModelEval(self,
                            numPredictions=numPredictions,
                            modelName=name,
                            singleEval=False,
                            modelPath=modelDir
                            )
      
      df = pd.concat([df, pd.DataFrame({'AJI': [ret[1]]}, index=[name])])
    
    df = df['AJI'].apply(lambda x: pd.Series(x).describe())

    timeElapsed = ((time() - self.timeStart) / 60).__round__(2)
    print(f"=== AJI Dataframe after {timeElapsed} minutes===", 
          df, '=== ===', sep='\n')

if __name__ == "__main__":
  x = Schmoo(model_dir='../../models',
            data_dir='../../data/tania', 
            predict_dir='../../predictions', 
            diam_mean=80
          )
  
  dataGen, train, test = False, False, False
  eval, compModel = False, True

  if dataGen: 
    ret = x.DataGenList('img')
    print(len(ret))
    for y in ret:
      print(type(y))

  if train: x.TrainModel(savePath='../..')

  if test: x.ModelPredict(model_name='cyto_lr20_wd20000_ep100_7559516', 
                          numPredictions=5,
                        )
    
  if eval: 
    ret = x.ModelEval(modelName='cyto_lr20_wd20000_ep100_7559516',
                      numPredictions=5,
                    )
    
    print(ret[1], f"Mean: {ret[1].mean()}", sep='\n')

  if compModel: x.CompareModels('../../saved_models', 14)