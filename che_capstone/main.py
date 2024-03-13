
from src.steady_state import SteadyState

class Master(SteadyState):
  def __init__(self, 
            targetFlow: int = 1, # mtpd
            targetCompound: str = "NH3",
            cfgPath: str = "./cfg.json"
          ) -> None:
    
    print(targetFlow, targetCompound)

    super().__init__(
              targetFlow=targetFlow, 
              targetCompound=targetCompound,
              cfgPath=cfgPath,
              maxIterations=15
            )
    
    Master.ssa(self)



if __name__ == "__main__":
  Master()