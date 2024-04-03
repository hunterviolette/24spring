
import os
import json
import numpy as np

from functools import reduce

if os.getcwd().split("/")[-1].endswith("src"):
  from core import SinglePass
else: 
  from src.core import SinglePass

class SteadyState(SinglePass):
  def __init__(self, 
            cfgPath: str = "./cfg.json",
            maxIterations: int = 25
          ) -> None:
    
    self.maxIter = maxIterations

    SinglePass.__init__(self, cfgPath)
    
  def Steady_State_Flow(self, excess:float=1):
    
    for iter in range(self.maxIter):
      SteadyState.IterFlows(self, True, iter, excess, True)
      SteadyState.FlowFeatures(self, iter)
      
      tuPath = ["Basis", "Overall Reaction", self.targetCompound, "unit"]
      targetUnit = reduce(lambda d, k: d[k], tuPath, self.c)

      pathing = ["Units", targetUnit, "flow", "products", self.targetCompound, self.cols["m"]]
      flow = self.q(reduce(lambda d, k: d[k], pathing, self.c), 'kg/batch')

      if iter>0:
        with open(f'states/iter_{iter-1}.json', "r") as f: ps = json.load(f)
        pflow = self.q(reduce(lambda d, k: d[k], pathing, ps), 'kg/batch')

        if flow.__round__(7) == pflow.__round__(7): 
          self.ssflow = flow
          break
      
      if iter >= self.maxIter: break
    
  def Steady_State_Setpoint(self, excess:float=1):

    for iter in range(self.maxIter):
      [os.remove(f"./states/{x}") for x in os.listdir('./states')]
      
      SteadyState.Steady_State_Flow(self, excess)
      if hasattr(self, 'ssflow'):
        print(f"{iter} Iteration converging steady-state flows at",
              f" {self.ssflow.to('mtpd').__round__(7)}")

        ssFlow = self.ssflow.to('mtpd').__round__(7) # calculated ss-flow
        spFlow = self.targetFlow.to('mtpd') # desired ss-flow

        if not np.isclose(ssFlow, spFlow, atol=0.0001 * spFlow.magnitude):
          excess = (spFlow/ssFlow).magnitude + .0005
        else: 
          print(f"Converged setpoint to target flow after {iter} iterations")
          break
      elif not hasattr(self, 'ssFlow') and iter>0: 
        print(f"Could not find steady-state after {self.maxIter} iterations")
        break
        
if __name__ == "__main__":
  SteadyState().Steady_State_Flow()


