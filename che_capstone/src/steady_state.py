
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

    SinglePass.__init__(self,
              targetFlow=targetFlow, 
              targetCompound=targetCompound,
              cfgPath=cfgPath
            )
    
  def SSA(self):
    
    for iter in range(self.maxIter):
      SteadyState.FlowFeatures(self, iter)

if __name__ == "__main__":
  SteadyState().ssa()


