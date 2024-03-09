import pandas as pd
import pint
import pprint
import json

from chempy import Substance

from v3_balance import Balance

class SinglePass(Balance):

  def __init__(self, 
              targetFlow: int = 1 # mtpd 
            ) -> None:
    
    super().__init__(targetflow=targetFlow)
    SinglePass.FractionalConversion(self)
    self.batchFlows = self.batchFlows.drop(
      columns=["Stoch Coeff", "Mass Flow (mtpd/20-min-batch)"])
    
    self.cols = {
          "mw": 'Molecular Weight (grams/mol)',
          "n": "Component Flow (kmol/20-min-batch)",
          "m": "Mass Flow (kg/20-min-batch)"
        }

  def main(self):
    cfg = self.c
    for stage in cfg["Stages"]:
      for unit in cfg["Stages"][stage]:
        uo = cfg["Units"][unit]
        uo.setdefault("flow", {}).setdefault("reagents", {})
        uo.setdefault("source", {})
        
        conv = min(float(uo["conversion"]), 1) if "conversion" in uo else 1
        if stage == str(0):
          for comp in [x for x in uo["reaction"]["reagents"].keys()
                  if x in cfg["initCompounds"]]:
            
            row = self.batchFlows.loc[
              (self.batchFlows["Component"] == comp) & 
              (self.batchFlows[self.cols["m"]] < 0)
              ]
            
            uo["flow"] = {
              "reagents": {
                  comp: {
                    self.cols["n"]: row[self.cols["n"]].values[0],
                    self.cols["m"]: row[self.cols["m"]].values[0],
                  }
                }
              }
          
        else:
          for comp in uo["reaction"]["reagents"]:
            for input in uo["inputs"]:
              inputuo = cfg["Units"][input]
              if comp in inputuo["flow"]["products"].keys():
                
                n = (inputuo["flow"]["products"][comp][self.cols["n"]]
                     * uo["inputs"][input][comp])
                m = inputuo["flow"]["products"][comp][self.cols["m"]]
                
                row = uo["flow"]["reagents"].get(comp, {})
                uo["flow"]["reagents"][comp] = {
                  self.cols["n"]: n + row.get(self.cols["n"], 0),
                  self.cols["m"]: m + row.get(self.cols["m"], 0)
                }
            
            for source in uo["source"]:
              if "limiting_reagent" in uo:
                reag = uo["limiting_reagent"]
                if reag in uo["flow"]["reagents"]:
                  
                  n = (uo["flow"]["reagents"][reag][self.cols["n"]]
                        * uo["reaction"]["reagents"][reag]
                        / uo["reaction"]["reagents"][source]
                      ).__abs__()
                  
                  m = (self.q(n, 'kmol/batch') * 
                        self.q(self.subs[source].mass, 'g/mol')
                      ).to("kg/batch")
                    
                  row = uo["flow"]["reagents"].get(comp, {})
                  uo["flow"]["reagents"][comp] = {
                      self.cols["n"]: n + row.get(self.cols["n"], 0),
                      self.cols["m"]: m.magnitude + row.get(self.cols["m"], 0)
                  }

                else: raise Exception("Limiting reagent not found")


        if uo["flow"]["reagents"].keys() == uo["reaction"]["reagents"].keys():
          
          if len(uo["reaction"]["reagents"].keys()) == 1: 
            reagent = next(iter(uo["reaction"]["reagents"]))
          else: reagent = uo["limiting_reagent"]

          for prod in uo["reaction"]["products"].keys():
            # Add conditional to use mass basis instead of mole if reagent is air
            
            stoich = (uo["reaction"]["products"][prod] / 
                      abs(uo["reaction"]["reagents"][reagent]))
            
            n = (stoich * conv * 
                uo["flow"]["reagents"][reagent][self.cols["n"]]
                ).__abs__()
            
            m = (self.q(n, 'kmol/batch') * 
                  self.q(self.subs[prod].mass, 'g/mol')
                  ).to("kg/batch")

            uo["flow"].setdefault("products", {}
              ).update({
                prod: {
                  self.cols["n"]: n,
                  self.cols["m"]: m.magnitude,
                }
              })
          
          if conv < 1:
            for reagent in uo["reaction"]["reagents"].keys():
              rflow = uo["flow"]["reagents"][reagent]

              uo["flow"]["products"][reagent] = {
                self.cols["n"]: rflow[self.cols["n"]] * (1 - conv),
                self.cols["m"]: rflow[self.cols["m"]] * (1 - conv),
              }

          pprint.pprint(f"=== Stage: {stage}, Unit: {unit} ===")
          pprint.pprint(self.c["Units"][unit])
        else: raise Exception(f"Missing reagents {stage} {unit}")

      if stage == str(1): break

if __name__ == "__main__":
  x = SinglePass()
  x.main()