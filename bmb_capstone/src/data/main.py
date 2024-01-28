from cellpose import models
from os import listdir

class Schmoo:
    
    def __init__(self, 
                 useGpu: bool = True,
                 model_dir: str = './models'
                ) -> None:
        
        if useGpu: self.device = 'gpu'
        else: self.device = 'cpu'
        
        self.model_dir = model_dir
    
    def DataGenerator(self, directory: str = './data/train'):
        for x in listdir(directory):
            print(x)
    
    def TrainModel(self,
                   model_type: str = 'cyto',
                   image_channels: list[int] = [2,1],
                   learning_rate: float = .1,
                   weight_decay: float = .0001,
                   n_epochs: int  = 100 
                ):

        images_train, masks_train = Schmoo.DataGenerator(self, './data/train')
        name = "_".join([model_type,
                        f"lr{int(learning_rate*100,0)}",
                        f"_wd{int(learning_rate*100000)}",
                        f"_ep{n_epochs}"
                    ])

        model = models.CellposeModel(device=self.device)
        model.train(images_train, masks_train, model_type=model_type, 
                    chan=image_channels, learning_rate=learning_rate, 
                    weight_decay=weight_decay, n_epochs=n_epochs
                )

        model.save_model(f"{self.model_dir}/{name}")

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
    x = Schmoo().DataGenerator()
