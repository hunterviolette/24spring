from tifffile import imread, imwrite
from skimage.io import imread as pngRead
from cellpose import models
from cellpose import metrics
from os.path import abspath, exists
from os import listdir, makedirs
from time import time
from datetime import datetime
from cv2 import resize
from typing import Optional

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

  def DataGenerator(self):
    data, data_labels, names = [], [], []
    for x in listdir(self.data_dir):
        if not "mask" in x:
            data.append(imread(f"{self.data_dir}/{x}"))
            try: data_labels.append(imread(f"{self.data_dir}/{x.replace('.', '_mask.')}"))
            except:
                y = x.replace('.', '_mask.').replace('.tif', '.png')
                try: data_labels.append(pngRead(f"{self.data_dir}/{y}"))
                except: raise Exception("Data generator error")
            names.append([x, x.replace('.', '_mask.')])

    assert len(data) == len(data_labels)
    
    print(f"img len: {len(data)}, mask len: {len(data_labels)}")
    return data, data_labels, names

  def TrainModel(self,
                model_type: str = 'cyto',
                image_channels: list[int] = [0,0], # gray scale images
                learning_rate: float = .2,
                weight_decay: float = .00001,
                n_epochs: int  = 100, 
                save_every: int = 10, # save after amount of epochs
                min_train_masks: int = 0,
                residual_on: bool = True, # celloose Unet if True
                rescale: bool = True,
                normalize: bool = True,
            ):

    print('=== init data generator ===')
    images_train, masks_train, file_train = Schmoo.DataGenerator(self)
    #images_test, masks_test, file_test = Schmoo.DataGenerator(self)
    print('=== returned data generator ===')

    name = "_".join([model_type,
                    f"lr{int(learning_rate*100)}",
                    f"wd{int(learning_rate*100000)}",
                    f"ep{n_epochs}", 
                    str(time()).split('.')[1]
                ])

    print(f'=== init training model: {name} ===')
    model = models.CellposeModel(gpu=self.gpu,
                                model_type=model_type,
                                diam_mean=self.diam_mean
                            )
    
    model.train(train_data=images_train, 
                train_labels=masks_train,
                #test_data=images_test,
                #test_labels=masks_test,
                channels=image_channels, 
                learning_rate=learning_rate, 
                weight_decay=weight_decay, 
                n_epochs=n_epochs,
                normalize=normalize,
                rescale=rescale,
                #save_every=save_every,
                min_train_masks=min_train_masks,
                save_path='./',
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
              stats: bool = False
          ):
      
    print('=== init data generator ===')
    images_test, masks_test, file_test = Schmoo.DataGenerator(self)
    print('=== returned data generator ===')

    if model_name in listdir(self.model_dir):
            
        model = models.CellposeModel(pretrained_model=f"{self.model_dir}/{model_name}",
                                    gpu=self.gpu,
                                )
        print(f'Opened model: {model_name}')

        stringTime = datetime.now().strftime('%Y-%m-%d_%H').replace('-', '_')
        savePath = f"{self.predict_dir}/{stringTime}_{model_name}"    

        if (len(images_test) > 0) and saveImages: 
          Schmoo.initDir(self, savePath)
          print(f"Saving images to: {savePath}")
            
        fig, stat = [], [] 
        for i, img in enumerate(images_test):
            filename = file_test[i][0]
            mask, flow, styles = model.eval(img,
                                            channels=image_channels,
                                            rescale=True,
                                            diameter=self.diam_mean)
            print(f"Evaluated: {filename}")
            
            if round(mask.mean(),5) > 0:
                if saveImages:
                  imwrite(f"{savePath}/img_{filename}", img)
                  imwrite(f"{savePath}/mask_{filename}", mask)
                  print(f"Saved img/mask to img_{filename} / mask_{filename}")
                
                if figures: 
                  fig.append([x for x in [resize(img, imgResize), resize(mask, imgResize), filename]])

                if stats: stat.append([mask, filename])

            else: print(f"No mask found for {filename}")

            if (numPredictions != None) and (i+1 >= numPredictions): break 
    else: raise Exception(f"{model_name}: not found in {self.model_dir}")
    if (len(fig) > 0) or (len(stat) > 0): 
        if figures and stats: 
          return [fig, stat]
        elif figures: return fig
        elif stats: return stat
    
if __name__ == "__main__":
  x = Schmoo(data_dir='./data/tania', diam_mean=80)
  dataGen, train, test = False, False, False
  debug = True

  if dataGen:
      data, data_labels, names = x.DataGenerator()
      print(data[0], data[0].mean(), 
          data_labels[0], data_labels[0].mean(),
          names[0], 
          sep='\n'
      )
  
  if train: x.TrainModel()
  if test: x.TestModel(model_name='cyto_lr20_wd20000_ep100_7559516', 
                      debug=debug)