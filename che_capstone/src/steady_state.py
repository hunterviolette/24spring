
import os
import json
import numpy as np
import pandas as pd

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
        
  def CloseBalance(self, debug: bool=True):
    flow = self.flows.loc[self.flows["Iteration"] == self.flows["Iteration"].max()]
    
    if debug:
      path = max(os.listdir("./states"), key=lambda x: int(x.split("_")[-1].split(".")[0]))
      with open(f"./states/{path}", "r") as f: cfg = json.load(f)
    else: cfg = self.c

    d = pd.DataFrame()
    for k, v in cfg["Basis"]["Overall Reaction"].items():

      flow = cfg["Units"][v["unit"]]["flow"]
      flowIn = flow["reagents"].setdefault(k, {}).setdefault(self.cols["n"], 0)
      flowOut = flow["products"].setdefault(k, {}).setdefault(self.cols["n"], 0)

      d = pd.concat([d, 
                    pd.DataFrame({
                      "Component": [k],
                      f"Stoich": [v["stoich"]],
                      f"Reacted {self.cols['n']}": [abs(flowIn - flowOut)],
                      f"Out {self.cols['n']}": [flowOut],
                      f"In {self.cols['n']}": [flowIn],
                    })])
    
    df = d.set_index('Component')
    for comp in df.index:
      stoich = abs(df.loc[comp, 'Stoich'])
      flow = df.loc[comp, 'Reacted Flow (kmol/batch)']
      
      for dcomp in [x for x in df.index if x != comp]:
        df.loc[comp, f'{dcomp} balance'] = (
            flow / stoich * abs(df.loc[dcomp, 'Stoich']) 
            - df.loc[dcomp, 'Reacted Flow (kmol/batch)']
          ).__round__(3)
            
    self.ssbal = df.drop(columns=["In Flow (kmol/batch)", "Out Flow (kmol/batch)"]
                  ).replace({0.000: True, -0.000: True})
    
    print('+'*10, flow, '+'*10, self.ssbal, sep='\n')

if __name__ == "__main__":
  SteadyState().Steady_State_Flow()


