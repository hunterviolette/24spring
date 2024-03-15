
from src.steady_state import SteadyState
from pages.util.util import Preprocessing

import pandas as pd

class Master(SteadyState, Preprocessing):
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
              maxIterations=4
            )
    
    Master.ssa(self)
    
    Master.UnitFlows(self)
    df = self.flows

    mdf = df.pivot_table(
                      index=['Unit', 'Iteration'], 
                      columns='Chemical', 
                      values='Flow (kmol/20-min-batch)', 
                      fill_value=0
                    )
    mdf.to_csv('iter_flows.csv')
    
    df = df.loc[df["Iteration"] == df["Iteration"].max()
              ].pivot_table(
                index=['Unit', 'Stream Type'], 
                columns='Chemical', 
                values='Flow (kmol/20-min-batch)', 
                fill_value=0
              )

    df.to_csv("stream_table.csv")




if __name__ == "__main__":
  Master()
  #df = pd.read_csv('flows.csv').set_index(["Iteration", "Unit"])