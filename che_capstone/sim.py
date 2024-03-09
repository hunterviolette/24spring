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
    
    #print(self.batchFlows)

    self.cols = {
          "mw": 'Molecular Weight (grams/mol)',
          "n": "Component Flow (kmol/20-min-batch)",
          "m": "Mass Flow (kg/20-min-batch)"
        }

  def main(self):
    cfg = self.c
    for stage in cfg["Stages"]:
      print(stage)
      for unit in cfg["Stages"][stage]:
        print(unit)
        if stage == str(0):
          uo = cfg["Units"][unit]
    
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
          
          conv = float(uo["conversion"]) # fractional converison
          if conv >1: conv=1 # no cheating

        if uo["reaction"]["reagents"].keys() == uo["flow"]["reagents"].keys():
          
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

          pprint.pprint(f"=== Unit: {unit} ===")
          pprint.pprint(self.c["Units"][unit])
      if stage ==1: break

if __name__ == "__main__":
  x = SinglePass()
  x.main()