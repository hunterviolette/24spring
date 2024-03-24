
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
      SteadyState.IterFlows(self, True, iter, excess)
      
      pathing = ["Units", "R-103", "flow", "products", self.targetCompound, self.cols["m"]]
      flow = self.q(reduce(lambda d, k: d[k], pathing, self.c), 'kg/batch')

      if iter>0:
        with open(f'states/iter_{iter-1}.json', "r") as f: ps = json.load(f)
        pflow = self.q(reduce(lambda d, k: d[k], pathing, ps), 'kg/batch')

        if flow.__round__(7) == pflow.__round__(7): 
          self.ssflow = flow
          SteadyState.FlowFeatures(self, iter)
          break
      
      if iter >= self.maxIter: break
    
  def Steady_State_Setpoint(self):

    for iter in range(self.maxIter):
      [os.remove(f"./states/{x}") for x in os.listdir('./states')]

      print(f"{iter} Iteration converging steady state flows at {self.targetFlow}")

      if iter == 0: SteadyState.Steady_State_Flow(self)
      else:
        ssFlow = self.ssflow.to('mtpd').__round__(3)
        if not np.isclose(ssFlow, self.targetFlow, atol=0.0001 * self.targetFlow):
          e = (self.targetFlow/ssFlow).magnitude + .0005
          SteadyState.Steady_State_Flow(self, excess=e)
        else:
          print(f"Converged setpoint to target flow after {iter} iterations")
          SteadyState.Steady_State_Flow(self, excess=e)
          break

      if iter >= self.maxIter: break
        
    print(self.ssflow.to('mtpd'), self.targetFlow)

if __name__ == "__main__":
  SteadyState().Steady_State_Flow()


