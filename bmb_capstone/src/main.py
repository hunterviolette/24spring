from tifffile import imread, imwrite
from cellpose import models
from cellpose import io
from os.path import abspath
from os import listdir
from time import time

class Schmoo:
    
    def __init__(self, 
                useGpu: bool = True,
                model_dir: str = './models'
                ) -> None:
        
        self.timeStart = time()
        self.gpu = useGpu        
        self.model_dir = model_dir
    
    def DataGenerator(self, directory: str = './data/training'):
        data, data_labels, names = [], [], []
        for x in listdir(directory):
            if not "mask" in x:
                data.append(imread(f"{directory}/{x}"))
                data_labels.append(imread(f"{directory}/{x.replace('.', '_mask.')}"))
                names.append([x, x.replace('.', '_mask.')])

        assert len(data) == len(data_labels)
        
        print(f"img len: {len(data)}", f"mask len: {len(data_labels)}", sep='\n')
        return data, data_labels, names

    def TrainModel(self,
                model_type: str = 'cyto',
                image_channels: list[int] = [2,1],
                learning_rate: float = .1,
                weight_decay: float = .0001,
                n_epochs: int  = 100, 
                save_every: int = 10, # save after amount of epochs
                min_train_masks: int = 0
            ):

        print('=== init data generator ===')
        images_train, masks_train, file_train = Schmoo.DataGenerator(self, './data/training')
        #images_test, masks_test, file_test = Schmoo.DataGenerator(self, './data/test')
        print('=== returned data generator ===')

        name = "_".join([model_type,
                        f"lr{int(learning_rate*100)}",
                        f"wd{int(learning_rate*100000)}",
                        f"ep{n_epochs}"
                    ])

        print(f'=== init training model: {name} ===')
        model = models.CellposeModel(gpu=False,
                                    model_type=model_type,
                                )
        
        model.train(train_data=images_train, 
                    train_labels=masks_train,
                    #test_data=images_test,
                    #test_labels=masks_test,
                    channels=image_channels, 
                    learning_rate=learning_rate, 
                    weight_decay=weight_decay, 
                    n_epochs=n_epochs,
                    #save_every=save_every,
                    min_train_masks=min_train_masks,
                    save_path='./',
                    model_name=name
                )
        
        print(f"=== saved {name} to ./ after {time()-self.timeStart} seconds ===")

    @staticmethod
    def ExportMask(images_test, masks, flows, file_test, model_name):

        io.save_masks(
                images=images_test, 
                masks=masks, 
                flows=flows, 
                file_names=f"{file_test}",
                savedir=abspath(f"./masks/{model_name.replace('./models/','')}")
            )

        print(f"Mask mean: {round(masks.mean(), 5)}", 
            f"Saved output to ./masks/{model_name.replace('./models/','')}/{file_test}",
            sep=' ')

    def TestModel(self,
                model_name: str = None,
                image_channels: list[int] = [2,1],
                debug: bool = False
            ):
        
        print('=== init data generator ===')
        images_test, masks_test, file_test = Schmoo.DataGenerator(self, './data/test')
        print('=== returned data generator ===')

        if model_name != None:
            if model_name in listdir(self.model_dir):
                
                model = models.CellposeModel(pretrained_model=f"{self.model_dir}/{model_name}",
                                            gpu=self.gpu,
                                        )
                print(f'Opened model: {model_name}')

                for i,x in enumerate(images_test):
                    masks, flows, styles = model.eval(images_test[0],
                                                    channels=image_channels,
                                                )
                    Schmoo.ExportMask(images_test[i], masks, flows, file_test[i][1], model_name)
                    if debug: break
            else:
                model = models.Cellpose(model_type=model_name,
                                        gpu=self.gpu,
                                    )
                print(f'Opened model: {model_name}')

                for i,x in enumerate(images_test):
                    masks, flows, styles, diams = model.eval(images_test[i],
                                                            channels=image_channels,
                                                )
                    Schmoo.ExportMask(images_test[i], masks, flows, file_test[i][1], model_name)
                    if debug: break
        
    def TestModels(self, image_channels: list[int] = [2,1]):
        images_test, masks_test, file_test = Schmoo.DataGenerator(self, './data/test')

        for i, x in enumerate(listdir(self.model_dir)):
            modelName = f"{self.model_dir}/{x}"

            print(i, modelName)
            if not x in listdir(self.model_dir):
                model = models.Cellpose(model_type=modelName,
                                        gpu=self.gpu,
                                        )
                
                print(f'Opened model: {modelName}')

                masks, flows, styles, diams = model.eval(images_test[0],
                                                        channels=image_channels,
                                            )
            else:
                model = models.CellposeModel(pretrained_model=modelName,
                                            gpu=self.gpu,
                                        )
                
                print(f'Opened model: {modelName}')

                masks, flows, styles = model.eval(images_test[0],
                                                channels=image_channels,
                                            )
            io.save_masks(
                    images=images_test[0], 
                    masks=masks, 
                    flows=flows, 
                    file_names=f"{file_test[i][1]}",
                    #savedir=f"./masks/{modelName.replace('./models/','')}"
                    savedir=abspath(f"./masks/{modelName.replace('./models/','')}")
                )
            
            print(masks.mean())
            print(f"Saved output to ./masks/{modelName.replace('./models/','')}/{file_test[i][1]}")
        
if __name__ == "__main__":
    x = Schmoo()
    dataGen, train, test = False, False, True

    if dataGen:
        data, data_labels, names = x.DataGenerator()
        print(data[0], data[0].mean(), 
            data_labels[0], data_labels[0].mean(),
            names[0], 
            sep='\n'
        )
    
    if train: x.TrainModel()
    if test: x.TestModel(model_name='cyto2torch_0')