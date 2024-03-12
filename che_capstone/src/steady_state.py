
import os

if os.getcwd().split("/")[-1].endswith("src"):
  from core import SinglePass
else: 
  from src.core import SinglePass

class SteadyState(SinglePass):
  def __init__(self, 
            targetFlow: int = 1, # mtpd
            targetCompound: str = "NH3",
            cfgPath: str = "./cfg.json",
            maxIterations: int = 50
          ) -> None:
    
    print(targetFlow, targetCompound)
    self.maxIter = maxIterations

    super().__init__(
              targetFlow=targetFlow, 
              targetCompound=targetCompound,
              cfgPath=cfgPath
            )
    
  def ssa(self):
    
    for iter in range(self.maxIter):
      SteadyState.IterFlows(self, True, iter)

if __name__ == "__main__":
  SteadyState().ssa()


