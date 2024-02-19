import numpy as np
import pandas as pd
import os

from time import time, sleep
from datetime import datetime
from typing import Optional, List, Union
from itertools import product
from functools import reduce
from math import ceil

from tifffile import imread, imwrite
from skimage.io import imread as pngRead
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import normalized_root_mse as nrmse

from cv2 import resize

from cellpose import models
from skimage.metrics import normalized_root_mse as nrmse
from skimage.metrics import structural_similarity as ssim
from cellpose.metrics import aggregated_jaccard_index, average_precision
from torch import load, save

if __name__ == '__main__':
    from util import Preprocessing
else:
    from pages.util.util import Preprocessing

class Schmoo(Preprocessing):
    
  def __init__(self, 
              useGpu: bool = True,
              model_dir: str = './vol/models',
              data_dir: str = './vol/image_data/tania',
              predict_dir: str = './vol/predictions',
              diam_mean: float = 30,
              ) -> None:
    
    self.timeStart = time()
    self.gpu = useGpu        
    self.model_dir = model_dir
    self.data_dir = data_dir
    self.predict_dir = predict_dir
    self.diam_mean = diam_mean

  def ElapsedTime(self):
    deltaT = ((time() - self.timeStart)).__round__(2)
    if deltaT > 60: return f"{(deltaT/60).__round__(2)} minutes"
    else: return f"{deltaT} seconds"
  
  def DataGenerator(self, 
                    maskRequired: bool = True,
                    directoryPath: Optional[str] = None,
                  ):
    
    self.dataGen = {}
    print('=== init data generator ===')
    if directoryPath == None: directory = self.data_dir
    else: directory = directoryPath
    
    for x in [f for f in os.listdir(directory) if f.endswith(('.tif', '.png'))]:
      if maskRequired:
        if not "mask" in x:
          img = imread(f"{directory}/{x}")
          
          try:  mask = imread(f"{directory}/{x.replace('.', '_mask.')}")
          except: 
            y = x.replace('.', '_mask.').replace('.tif', '.png')
            
            try: mask = pngRead(f"{directory}/{y}")
            except: raise Exception(f"Mask file not found for {x}")

          self.dataGen[x] = {'img':img, 'mask':mask, 'pmask': None}
      else:
        if not "mask" in x:
          img = imread(f"{directory}/{x}")
          self.dataGen[x] = {'img':img, 'mask':None, 'pmask': None}
    
    print('=== returned data generator ===')

  def BigDataGenerator(self, 
                      directories: List[Union[str, List[str]]],
                      maskRequired: bool = True
                    ):
    
    if isinstance(directories, str): directories = [directories]
    
    self.bigGen = {}
    for directory in directories:
      Schmoo.DataGenerator(self, maskRequired, str(directory))
      self.bigGen.update(self.dataGen)
    
  def DataGenList(self, 
                  listType: str # Takes img, mask, pmask
                ):
    if not hasattr(self, 'dataGen'): Schmoo.DataGenerator(self)    

    if listType in ["img", "mask", "pmask"]:
      return list(map(lambda key: self.dataGen[key][listType], self.dataGen.keys()))
    else: raise Exception("listType only accepts img, mask, pmask")

  def DataGenInstance(self,
                      maskRequired: bool = True,
                      bigGen: Optional[Union[str, List[str], None]] = None,
                    ):
    
    if bigGen != None: Schmoo.BigDataGenerator(self, bigGen, maskRequired)

    if hasattr(self, 'bigGen'): self.dataGen = self.bigGen

    if not hasattr(self, 'dataGen'): Schmoo.DataGenerator(self, maskRequired)

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
            savePath: str = './',
            modelName: Optional[str] = None
          ):

    Schmoo.DataGenInstance(self)
    
    if not hasattr(self, 'cellposeModel'):
      self.cellposeModel = models.CellposeModel(
                                    gpu=self.gpu,
                                    model_type=model_type,
                                    diam_mean=self.diam_mean
                                  )

    print(f'=== init training model: {modelName} ===')
    self.cellposeModel.train(
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
                          model_name=modelName
                        )

    print(f"=== saved {modelName} to ./ after {Schmoo.ElapsedTime(self)} ===")
    return modelName

  def BatchTrain(self, 
                savePath: str = './models',
                steps: int = 3,
                sEpoch: int = 200,
                eEpoch: int = 600,
                sLearningRate: float = .2,
                eLearningRate: float = .6,
                sWeightDecay: float = .0001,
                eWeightDecay: float = .01,
                diamMeans: List[int] = [30],
                baseModels: List[str] = ['cyto', 'cyto2'],
                dataPaths: List[Union[str, List[str]]] = ['./data/original'],
                train: bool = False
              ):
    
    learningRates = np.linspace(sLearningRate, eLearningRate, steps)
    weightDecays = np.linspace(sWeightDecay, eWeightDecay, steps)
    numEpochs = list(range(sEpoch, eEpoch+1, ceil(eEpoch/steps)))

    params = [dataPaths, 
              learningRates, 
              weightDecays, 
              numEpochs,
              baseModels,
              diamMeans,
            ]    

    totalCount = reduce(lambda x,y: x*y,[len(param) for param in params])
    if not train: 
      print(f'These parameters will train: {totalCount} models')
      return 
    
    print("\n".join([f"=== This will train {totalCount} models,", 
                    f"ctrl+C terminal within 5 seconds to cancel ==="]))
    sleep(5)
    print("=== Init batch training ===")
    
    names = []
    for dpaths in dataPaths:
      Schmoo.BigDataGenerator(self, dpaths, True)
      for baseModel in baseModels:
        for diamMean in diamMeans: 
          self.cellposeModel = models.CellposeModel(
                                          gpu=self.gpu,
                                          model_type=baseModel,
                                          diam_mean=diamMean
                                        )

          for dataPath, learningRate, weightDecay, numEpoch in product(
              dataPaths, learningRates, weightDecays, numEpochs):

            name = "_".join([baseModel,
                            f"lr{int(learningRate*100)}",
                            f"wd{int(weightDecay*100000)}",
                            f"ep{numEpoch}", 
                            f"dm{diamMean}",
                            str(time()).split('.')[1]
                        ])

            print(f"Model {len(names)+1}/{totalCount}", 
                  dataPath, baseModel, 
                  learningRate, weightDecay, 
                  numEpoch, sep=', ')
            
            names.append(name)
            
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
              modelName = name
            )

        
      if savePath != './models':
        for name in names:
          save(load(f"{savePath}/models/{name}"), f"{savePath}/{name}")
          os.remove(f"{savePath}/models/{name}")
        os.rmdir(f"{savePath}/models")
        
      print(f'=== Finished training after {Schmoo.ElapsedTime(self)}')

  def Predict(self,
              model_name: str = 'cyto2torch_0',
              image_channels: list[int] = [0,0], # Assumes 2 channel greyscale
              numPredictions: Optional[int] = None, # Number of predictions before breaking loop
              imgResize: Optional[int] = None, # int value in pixels  
              saveImages: bool = False,
              figures: bool = True, # Return list of images 
              hasTruth: bool = False, # Has true masks for images 
              modelPath: Optional[str] = None,
              diamMean: Optional[int] = None,
            ):
    
    Schmoo.DataGenInstance(self, maskRequired=hasTruth)

    if not isinstance(modelPath, str): dir = self.model_dir
    else: dir = modelPath

    if isinstance(diamMean, int): diam = diamMean
    else: diam = self.diam_mean

    if model_name in os.listdir(dir):
            
      if not hasattr(self, 'cellposeModel'):
        print("no attribute for self.cellposeModel, creating")
        self.cellposeModel = models.CellposeModel(
                                      gpu=self.gpu,
                                      pretrained_model=f"{dir}/{model_name}",
                                    )
      
      print(f'Opened model: {model_name}')

      stringTime = datetime.now().strftime('%Y-%m-%d_%H').replace('-', '_')
      savePath = f"{self.predict_dir}/{stringTime}"    

      if (len(self.dataGen.keys()) > 0) and saveImages: 
        Schmoo.initDir(savePath)
        print(f"Saving images to: {savePath}")
          
      fig, count = [], 0 
      for key in self.dataGen.keys():
        row = self.dataGen[key]
        mask, img = row["mask"], row["img"]

        pmask, flow, styles = self.cellposeModel.eval(
                                                  img,
                                                  channels=image_channels,
                                                  rescale=True,
                                                  diameter=diam
                                                )
          
        print(f"Generated mask for {key}, with mean of {round(np.mean(pmask),2)}")
        self.dataGen[key]["pmask"] = pmask
                
        if saveImages:
          imwrite(f"{savePath}/{key.replace('.', '_mask.')}", pmask)
          print(f"Saved mask for {key}")
        
        if figures: 
          if isinstance(imgResize, int):
            if imgResize == 0: pass
            else:
              tupleSize = (imgResize, imgResize)
              img = resize(img, tupleSize)
              pmask = resize(pmask, tupleSize)
              if hasTruth: mask = resize(mask, tupleSize)

          if hasTruth: fig.append([img, pmask, key, 
                                  Schmoo.Eval(self, 
                                              predMask=self.dataGen[key]["pmask"], 
                                              trueMask=self.dataGen[key]["mask"], 
                                              singleEval=True
                                            ),
                                  mask
                                          ], )
            
          else: fig.append([img, pmask, key])

        count += 1
        if (numPredictions != None) and (count >= numPredictions): break 

      if len(fig) > 0: return fig

    else: 
      raise Exception(f"{model_name}: not found in {dir}")
  
  def Eval(self, 
          predMask: Optional[np.ndarray] = None,
          trueMask: Optional[np.ndarray] = None,
          numPredictions: Optional[int] = None,
          diamMean: Optional[int] = None,
          modelName: Optional[str] = 'cyto_lr20_wd20000_ep100_7559516',
          singleEval: bool = False,
          modelPath: Optional[str] = None,
        ):

    Schmoo.DataGenInstance(self)

    if not singleEval:
      Schmoo.Predict(self, 
                    model_name=modelName,
                    numPredictions=numPredictions,
                    modelPath=modelPath,
                    diamMean=diamMean,
                  )
      
      self.dataGen = {key: value for key, value in self.dataGen.items() 
                      if isinstance(value['pmask'], np.ndarray) and
                        isinstance(value['mask'], np.ndarray)
                    }
      
      
      df = pd.DataFrame()
      for key in self.dataGen:
        row = self.dataGen[key]
        aji = aggregated_jaccard_index([row["mask"]], [row["pmask"]])
        ap = average_precision([row["mask"]], [row["pmask"]], threshold=[.5])
        
        mask = Schmoo.DesegmentMask(row["mask"])
        pmask = Schmoo.DesegmentMask(row["pmask"])
        
        df = pd.concat([df,
                        pd.DataFrame({
                          "model": [modelName],
                          "key": [key],
                          "euclidean normalized rmse": [nrmse(mask, pmask)],
                          "structural similarity": [ssim(mask, pmask)],
                          "jaccard index": [aji[0]],
                          "average precision": [ap[0][0][0]],
                          "true positives": [ap[1][0][0]],
                          "false positives": [ap[2][0][0]],
                          "false negatives": [ap[3][0][0]],
                        })], axis=0, ignore_index=True)
      return df
    else: 
      if not isinstance(trueMask, np.ndarray) and \
        not isinstance(predMask, np.ndarray):
          return 0
      
      return aggregated_jaccard_index(
                  [Schmoo.DesegmentMask(trueMask)], 
                  [Schmoo.DesegmentMask(predMask)]
                )[0]

  def BatchEval(self, 
                modelDir: str = './vol/models',
                numPredictions: Optional[int] = None,
                diamMeans: Union[List[int], int] = [30, 80],
                saveCsv: bool = False,
                testModels: List[str] = None,
                imageDir: Union[str, List[str], None] = None
              ):
    
    if isinstance(diamMeans, int): diamMeans = [diamMeans]
    
    Schmoo.DataGenInstance(self, maskRequired=True, bigGen=imageDir)
    
    modelList = [f for f in os.listdir(modelDir) 
                if not os.path.isdir(f"{modelDir}/{f}")]
    
    if isinstance(testModels, list):
      for tmodels in testModels:     
        modelList.extend(
              [f"{tmodels}/{f}" for f in os.listdir(f"{modelDir}"/{tmodels}) 
              if not os.path.isdir(f"{modelDir}/{tmodels}/{f}")]
            )
    
    df = pd.DataFrame()
    for name, diamMean in product(modelList, diamMeans):

      self.cellposeModel = models.CellposeModel(
                                    gpu=self.gpu,
                                    pretrained_model=f"{modelDir}/{name}",
                                  )
      
      print(f"Setting self.cellposeModel to {diamMean} {name}")
      
      d = Schmoo.Eval(self,
                      numPredictions=numPredictions,
                      modelName=name,
                      singleEval=False,
                      modelPath=modelDir,
                      diamMean=diamMean
                    )
    
      d["diam mean"] = diamMean
      df = pd.concat([df, d], axis=0, ignore_index=True)
    
    if saveCsv: 
      df.to_csv(f"{modelDir.split('/')[-1]}_{str(time()).split('.')[1]}.csv")

    timeElapsed = ((time() - self.timeStart) / 60).__round__(2)
    print(f"=== AJI Dataframe after {timeElapsed} minutes===", 
          df, '=== ===', sep='\n')
    
    return df

  def BatchLoop(self, modelDir: str = '../models/original_30diam_3step'):
    
    Schmoo.BatchTrain(self,
                      savePath=modelDir, 
                      steps=1,
                      train=True,
                    )
    
    Schmoo.BatchEval(self,
                    modelDir=modelDir, 
                    numPredictions=3, 
                    saveCsv=True,
                  )

if __name__ == "__main__":
  x = Schmoo(
          model_dir='../../../vol/models',
          data_dir='../../../vol/image_data/tania', 
          predict_dir='../../../vol/predictions', 
          diam_mean=30
        )
  
  dataGen, train, predict = False, False, False
  eval, batchEval, batchTrain = False, True, False

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

  if batchEval: x.BatchEval(modelDir='../../../vol/models', 
                            numPredictions=3, 
                            saveCsv=True,
                          )

  if batchTrain: x.BatchTrain(savePath='../../vol/models/original_30diam_3step', 
                              steps=1,
                              train=True)
    
