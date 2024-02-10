import numpy as np
import pandas as pd
import os

from time import time, sleep
from datetime import datetime
from typing import Optional
from itertools import product
from functools import reduce
from math import ceil

from tifffile import imread, imwrite
from skimage.io import imread as pngRead
from cv2 import resize

from cellpose import models
from cellpose.metrics import aggregated_jaccard_index
from torch import load, save

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

  def ElapsedTime(self):
    deltaT = ((time() - self.timeStart)).__round__(2)
    if deltaT > 60: return f"{(deltaT/60).__round__(2)} minutes"
    else: return f"{deltaT} seconds"
  
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

  def Train(self,
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
                    ).train(
                          train_data=Schmoo.DataGenList(self, 'img'), 
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

    print(f"=== saved {name} to ./ after {Schmoo.ElapsedTime(self)} ===")
    return name

  def Predict(self,
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
        img = self.dataGen[key]["img"]
        mask = self.dataGen[key]["mask"]

        pmask, flow, styles = model.eval(img,
                                        channels=image_channels,
                                        rescale=True,
                                        diameter=self.diam_mean
                                      )
        
        print(f"Generated mask for {key}")
        self.dataGen[key]["pmask"] = pmask
                
        if saveImages:
          imwrite(f"{savePath}/img_{key}", img)
          imwrite(f"{savePath}/mask_{key}", pmask)
          print(f"Saved img/mask for {key}")
        
        if figures: 
          if isinstance(imgResize, int):
            if imgResize == 0: pass
            else:
              tupleSize = (imgResize, imgResize)
              img, pmask = resize(img, tupleSize), resize(pmask, tupleSize)

          if hasTruth: fig.append([img, pmask, key, 
                                  Schmoo.Eval(self, pmask, mask, singleEval=True)])
            
          else: fig.append([img, pmask, key])

        count += 1
        if (numPredictions != None) and (count >= numPredictions): break 

      if len(fig) > 0: return fig

    else: raise Exception(f"{model_name}: not found in {dir}")
  
  def Eval(self, 
          predMask: Optional[np.ndarray] = None,
          trueMask: Optional[np.ndarray] = None,
          numPredictions: Optional[int] = None,
          modelName: Optional[str] = 'cyto_lr20_wd20000_ep100_7559516',
          singleEval: bool = False,
          modelPath: Optional[str] = None,
        ):

    if not hasattr(self, 'dataGen'): Schmoo.DataGenerator(self)

    if not singleEval:
      Schmoo.Predict(self, 
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

  def BatchTrain(self, 
                savePath: str = './models',
                steps: int = 2,
                sEpoch: int = 50,
                eEpoch: int = 500,
                sLearningRate: float = .001,
                eLearningRate: float = .1,
                sWeightDecay: float = .0001,
                eWeightDecay: float = .01,
              ):
    
    baseModels = ['cyto']
    dataPaths = [Schmoo.DataGenerator(self)]
    learningRates = np.linspace(sLearningRate, eLearningRate, steps)
    weightDecays = np.linspace(sWeightDecay, eWeightDecay, steps)
    numEpochs = list(range(sEpoch, eEpoch+1, ceil(eEpoch/steps)))

    params = [dataPaths, 
              baseModels, 
              learningRates, 
              weightDecays, 
              numEpochs,
            ]    

    totalCount = reduce(lambda x,y: x*y,[len(param) for param in params])
    
    print(f"=== This will train {totalCount+1} models, 5 seconds to cancel ===")
    sleep(5)
    print("=== Init batch training ===")

    names = []
    for dataPath, baseModel, learningRate, weightDecay, numEpoch in product(*params):
      print(f"Model {len(names)}/{totalCount}", 
            dataPath, baseModel, 
            learningRate, weightDecay, 
            numEpoch, sep=', ')
      
      names.append(
                Schmoo.Train(
                  self,
                  model_type=baseModel,
                  learning_rate=learningRate,
                  weight_decay=weightDecay,
                  n_epochs=numEpoch,
                  save_every=10000,
                  residual_on = True,
                  rescale = True,
                  normalize = True,
                  savePath = savePath,
                )
              )
    
    if savePath != './models':
      for name in names:
        save(load(f"{savePath}/models/{name}"), f"{savePath}/{name}")
        os.remove(f"{savePath}/models/{name}")
      os.rmdir(f"{savePath}/models")
      
    print(f'=== Finished training after {Schmoo.ElapsedTime(self)}')

  def BatchEval(self, 
                modelDir: str = './saved_models',
                numPredictions: Optional[int] = None,
              ):
    
    if not hasattr(self, 'dataGen'): Schmoo.DataGenerator(self)
    
    df = pd.DataFrame()
    print(os.listdir(modelDir))
    for name in os.listdir(modelDir):
      ret = Schmoo.Eval(self,
                        numPredictions=numPredictions,
                        modelName=name,
                        singleEval=False,
                        modelPath=modelDir
                      )
      
      df = pd.concat([df, pd.DataFrame({'AJI': [ret[1]]}, index=[name])])
    
    df = df['AJI'].apply(lambda x: pd.Series(x).describe())

    timeElapsed = ((time() - self.timeStart) / 60).__round__(2)
    print(f"=== AJI Dataframe after {timeElapsed} minutes===", 
          df.sort_values('mean', ascending=False), '=== ===', sep='\n')

if __name__ == "__main__":
  x = Schmoo(model_dir='../../models',
            data_dir='../../data/tania', 
            predict_dir='../../predictions', 
            diam_mean=80
          )
  
  dataGen, train, predict = False, False, False
  eval, batchEval, batchTrain = False, False, True

  if dataGen: 
    ret = x.DataGenList('img')
    print(len(ret))
    for y in ret:
      print(type(y))

  if train: x.Train(savePath='../..')

  if predict: x.Predict(model_name='cyto_lr20_wd20000_ep100_7559516', 
                          numPredictions=5,
                        )
    
  if eval: 
    ret = x.Eval(modelName='cyto_lr20_wd20000_ep100_7559516',
                numPredictions=5,
              )
    
    print(ret[1], f"Mean: {ret[1].mean()}", sep='\n')

  if batchEval: x.BatchEval('../../saved_models', 3)

  if batchTrain: x.BatchTrain(savePath='../../models/opt1', steps=3)