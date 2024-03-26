
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

  def main(self):
    Master.Steady_State_Setpoint(self)
    Master.UnitFlowsV2(self)
    
    if self.tables: 

      Master.Df_To_HTML(
          self.mb.drop(columns=["Mass Flow (kg/h)",
                                "Molecular Weight (grams/mol)", 
                                "Component Flow (kmol/h)"]),
          header="",
          name="./assets/non_ss_balance"
        )
      
      Master.CloseBalance(self, debug=False)
      df = self.flows.assign(**{
                "Flow (mtpd)": lambda x: x["Flow (kg/batch)"
                  ].apply(lambda y: self.q(y, 'kg/batch').to('mtpd').magnitude),
                
                "Stream Type": lambda x: x["Stream Type"
                  ].replace({"products": "outlet", "reagents": "inlet"})
                })
            
      Master.Df_To_HTML(
          self.ssbal,
          header="",
          name="./assets/ss_balance"
        )
      
      Master.Df_To_HTML(
          df,
          header="",
          name="./assets/iteration_table"
        )
      
      Master.Df_To_HTML(
          df.loc[
              (df["Iteration"] == df["Iteration"].max()) 
            ].pivot_table(
              index=['Unit', 'Iteration', "Stream Type"], 
              columns='Chemical', 
              values='Flow (kg/batch)', 
              fill_value=""
          ), 
          header="Inlet/Outlet flows for unit operations (kg/batch) at steady state",
          name="./assets/ss_pass"
        )

      Master.Df_To_HTML(
          df.loc[
              (df["Iteration"] == df["Iteration"].min()) 
            ].pivot_table(
              index=['Unit', 'Iteration', "Stream Type"], 
              columns='Chemical', 
              values='Flow (kg/batch)', 
              fill_value=""
          ), 
          header="Inlet/Outlet flows for unit operations (kg/batch) at first pass",
          name="./assets/s0_pass"
        )
      
      Master.Df_To_HTML(
          df.loc[
              (df["Iteration"] == df["Iteration"].max()) &
              (df["Stream Type"] != "inlet")
            ].pivot_table(
                index=['Unit', 'Stream Type'], 
                columns='Chemical', 
                values='Flow (mtpd)', 
                fill_value="", 
          ), 
          header="Inlet flows for unit operations (mtpd) at steady state",
          name="./assets/outlet_ss"
        )

      Master.Df_To_HTML(
          df.loc[
              (df["Iteration"] == df["Iteration"].max()) &
              (df["Stream Type"] == "inlet")
            ].pivot_table(
                index=['Unit', 'Stream Type'], 
                columns='Chemical', 
                values='Flow (mtpd)', 
                fill_value="", 
          ), 
          header="Outlet flows for unit operations (mtpd) at steady state",
          name="./assets/inlet_ss"
        )

if __name__ == "__main__":
  
  Master(tables=True).main()
