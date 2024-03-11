
from src.core import SinglePass

class Master(SinglePass):
  def __init__(self, 
            targetFlow: int = 1, # mtpd
            targetCompound: str = "NH3",
            cfgPath: str = "./cfg.json"
          ) -> None:
    
    print(targetFlow, targetCompound)

    super().__init__(
              targetFlow=targetFlow, 
              targetCompound=targetCompound,
              cfgPath=cfgPath
            )
    
    Master.features(self)



if __name__ == "__main__":
  Master()