
from src.steady_state import SteadyState
from pages.util.util import Preprocessing

import pandas as pd

class Master(SteadyState, Preprocessing):
  def __init__(self, 
            targetFlow: int = 1, # mtpd
            targetCompound: str = "NH3",
            cfgPath: str = "./cfg.json",
            tables: bool = False,
            maxIterations: int = 8
          ) -> None:
    
    print(targetFlow, targetCompound)

    SteadyState.__init__(self,
              targetFlow=targetFlow, 
              targetCompound=targetCompound,
              cfgPath=cfgPath,
              maxIterations=maxIterations
            )
    
    Master.SSA(self)
    
    if tables: 
      Master.UnitFlows(self)
      df = self.flows

      Master.Df_To_HTML(df.pivot_table(
                          index=['Unit', 'Iteration', "Stream Type"], 
                          columns='Chemical', 
                          values='Flow (kmol/20-min-batch)', 
                          fill_value=0
                          ), 
          header="Input and output flows for unit operations (kmol/20-min-batch) per iteration",
                        name="./assets/stream_iteration_table")
      
      Master.Df_To_HTML(df.loc[df["Iteration"] == df["Iteration"].max()
                            ].pivot_table(
                              index=['Unit', 'Stream Type'], 
                              columns='Chemical', 
                              values='Flow (kmol/20-min-batch)', 
                              fill_value=""
                          ), 
          header="Input and output flows for unit operations (kmol/20-min-batch) at steady state",
                        name="./assets/stream_table_steady_state")

if __name__ == "__main__":
  Master(tables=True)
