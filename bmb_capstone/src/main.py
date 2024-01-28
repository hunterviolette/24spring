from tifffile import imread, imwrite
from cellpose import models
from os import listdir

class Schmoo:
    
    def __init__(self, 
                useGpu: bool = True,
                model_dir: str = './models'
                ) -> None:
        
        if useGpu: self.device = 'cuda'
        else: self.device = 'cpu'
        
        self.model_dir = model_dir
    
    def DataGenerator(self, directory: str = './data/training'):
        data, data_labels = [], []
        for x in listdir(directory):
            if not "mask" in x:
                data.append(imread(f"{directory}/{x}"))
                data_labels.append(imread(f"{directory}/{x.replace('.', '_mask.')}"))

        assert len(data) == len(data_labels)
        
        print(f"img len: {len(data)}", f"mask len: {len(data_labels)}", sep='\n')
        return data, data_labels

    
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
        images_train, masks_train = Schmoo.DataGenerator(self, './data/training')
        print('=== returned data generator ===')

        name = "_".join([model_type,
                        f"lr{int(learning_rate*100)}",
                        f"wd{int(learning_rate*100000)}",
                        f"ep{n_epochs}"
                    ])

        print(f'=== init training model: {name} ===')
        model = models.CellposeModel(model_type=model_type)#device=self.device)
        model.train(images_train, 
                    masks_train, 
                    channels=image_channels, 
                    learning_rate=learning_rate, 
                    weight_decay=weight_decay, 
                    n_epochs=n_epochs,
                    save_every=save_every,
                    min_train_masks=min_train_masks,
                    save_path=self.model_dir,
                    model_name=name
                )
        
        print(f"=== saved {name} to {self.model_dir} ===")

    def TestModel(self,
                model_name: str = ''
            ):
        
        images_test, masks_test = Schmoo.DataGenerator(self, './data/test')
        
        # Load the model from disk
        model = models.CellposeModel(device=self.device).load_model()
        metrics = model.evaluate(images_test, masks_test)

        # Print evaluation metrics
        print("Evaluation metrics:")
        print(metrics)
        
if __name__ == "__main__":
    x = Schmoo().TrainModel()
