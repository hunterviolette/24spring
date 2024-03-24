
from src.steady_state import SteadyState
from pages.util.util import Preprocessing

import pandas as pd

class Master(SteadyState, Preprocessing):
  def __init__(self, 
            cfgPath: str = "./cfg.json",
            tables: bool = False,
            maxIterations: int = 100
          ) -> None:
    
    SteadyState.__init__(self,
              cfgPath=cfgPath,
              maxIterations=maxIterations
            )
    
    Master.Steady_State_Setpoint(self)
    
    if tables: 
      Master.UnitFlows(self)
      df = self.flows.astype({"Stream Type": 'string'})
      df["Stream Type"] = df["Stream Type"].replace({"products": "outlet", "reagents": "inlet"})

      Master.Df_To_HTML(df.pivot_table(
                          index=['Unit', 'Iteration', "Stream Type"], 
                          columns='Chemical', 
                          values='Flow (kmol/batch)', 
                          fill_value=0
                          ), 
          header="Input and output flows for unit operations (kmol/20-min-batch) per iteration",
                        name="./assets/stream_iteration_table")
      
      Master.Df_To_HTML(df.loc[df["Iteration"] == df["Iteration"].max()
                            ].pivot_table(
                              index=['Unit', 'Stream Type'], 
                              columns='Chemical', 
                              values='Flow (kmol/batch)', 
                              fill_value=""
                          ), 
          header="Input and output flows for unit operations (kmol/20-min-batch) at steady state",
                        name="./assets/stream_table_steady_state")
      
      Master.Df_To_HTML(df.loc[df["Iteration"] == df["Iteration"].max()].pivot_table(
                        index=['Unit', 'Stream Type'], 
                        columns='Chemical', 
                        values='Flow (kmol/batch)', 
                        fill_value="", 
                        aggfunc=lambda x: x * 3
                      ), 
          header="Input and output flows for unit operations (kmol/hour) at steady state",
                        name="./assets/stream_table_steady_state_hour")
      
      Master.Df_To_HTML(df.loc[(df["Iteration"] == df["Iteration"].max()) &
                              (df["Stream Type"] != "inlet")
                            ].pivot_table(
                                index=['Unit', 'Stream Type'], 
                                columns='Chemical', 
                                values='Flow (kmol/batch)', 
                                fill_value="", 
                                aggfunc=lambda x: x * 3
                      ), 
          header="Input and output flows for unit operations (kmol/hour) at steady state",
                        name="./assets/inlet_ss_hourly_hour")

      Master.Df_To_HTML(df.loc[(df["Iteration"] == df["Iteration"].max()) &
                              (df["Stream Type"] != "outlet")
                            ].pivot_table(
                                index=['Unit', 'Stream Type'], 
                                columns='Chemical', 
                                values='Flow (kmol/batch)', 
                                fill_value="", 
                                aggfunc=lambda x: x * 3
                          ), 
          header="Input and output flows for unit operations (kmol/hour) at steady state",
                        name="./assets/outlet_ss_hourly_hour")

if __name__ == "__main__":
  
  Master(tables=True)
