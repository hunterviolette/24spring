
from src.core import SinglePass

class Master(SinglePass):
  def __init__(self, 
            targetFlow: int = 1, # mtpd
            targetCompound: str = "NH3"  
          ) -> None:
    
    print(targetFlow, targetCompound)

    super().__init__(
              targetFlow=targetFlow, 
              targetCompound=targetCompound
            )
    
    Master.features(self)

if __name__ == "__main__":
  Master()