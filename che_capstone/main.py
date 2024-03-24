
from src.steady_state import SteadyState
from pages.util.util import Preprocessing

import pandas as pd
import json
import os

class Master(SteadyState, Preprocessing):
  def __init__(self, 
            cfgPath: str = "./cfg.json",
            tables: bool = False,
            maxIterations: int = 10
          ) -> None:
    
    self.tables = tables
    SteadyState.__init__(self,
              cfgPath=cfgPath,
              maxIterations=maxIterations
            )
  
  def SS_Balance(self, debug: bool=True):
    Master.UnitFlowsV2(self)
    df = self.flows.copy()
    df = df.loc[df["Iteration"] == df["Iteration"].max()
              ].assign(**{
                "Flow (mtpd)": lambda x: x["Flow (kg/batch)"
                  ].apply(lambda y: self.q(y, 'kg/batch').to('mtpd').magnitude),
                
                "Stream Type": lambda x: x["Stream Type"
                  ].replace({"products": "outlet", "reagents": "inlet"})
                })
    
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
    
    print('+++++++++', df, d, sep='\n')

  def main(self):
    Master.Steady_State_Setpoint(self)
    Master.SS_Balance(self, debug=False)

    if self.tables: 
      Master.UnitFlows(self)
      df = (self.flows
                .astype({"Stream Type": 'string'})
                .assign(**{
                        "Flow (mtpd)": lambda x: x["Flow (kg/batch)"
                          ].apply(lambda y: self.q(y, 'kg/batch').to('mtpd').magnitude),
                        
                        "Stream Type": lambda x: x["Stream Type"
                          ].replace({"products": "outlet", "reagents": "inlet"})
                        })
                      )
      
      print(df)
      Master.Df_To_HTML(df.pivot_table(
                          index=['Unit', 'Iteration', "Stream Type"], 
                          columns='Chemical', 
                          values='Flow (kg/batch)', 
                          fill_value=0
                          ), 
          header="Inlet/Outlet flows for unit operations (kmol/batch) per iteration",
                        name="./assets/stream_iteration_table")
      
      Master.Df_To_HTML(df.loc[df["Iteration"] == df["Iteration"].max()
                      ].pivot_table(
                        index=['Unit', 'Stream Type'], 
                        columns='Chemical', 
                        values='Flow (mtpd)', 
                        fill_value="", 
                      ), 
          header="Inlet/Outlet flows for unit operations (mtpd) at steady state",
                        name="./assets/ss_mtpd")
      
      Master.Df_To_HTML(df.loc[(df["Iteration"] == df["Iteration"].max()) &
                              (df["Stream Type"] != "inlet")
                            ].pivot_table(
                                index=['Unit', 'Stream Type'], 
                                columns='Chemical', 
                                values='Flow (kg/batch)', 
                                fill_value="", 
                                aggfunc=lambda x: x * self.bph
                      ), 
          header="Inlet flows for unit operations (kg/hour) at steady state",
                        name="./assets/outlet_ss_hourly")

      Master.Df_To_HTML(df.loc[(df["Iteration"] == df["Iteration"].max()) &
                              (df["Stream Type"] == "inlet")
                            ].pivot_table(
                                index=['Unit', 'Stream Type'], 
                                columns='Chemical', 
                                values='Flow (mtpd)', 
                                fill_value="", 
                                aggfunc=lambda x: x * self.bph
                          ), 
          header="Outlet flows for unit operations (kg/hour) at steady state",
                        name="./assets/inlet_ss_hourly")

if __name__ == "__main__":
  
  Master(tables=True).main()
