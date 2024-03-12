import pandas as pd
import pprint
import json
import os

from typing import Optional

if os.getcwd().split("/")[-1].endswith("src"):
  from balance import Balance
  from thermal import Therm
else: 
  from src.balance import Balance
  from src.thermal import Therm

class SinglePass(Balance, Therm):

  def __init__(self, 
              targetFlow: int = 1, # mtpd
              targetCompound: str = "NH3",
              cfgPath: str = "../cfg.json"
            ) -> None:
    
    self.cPath = cfgPath
    Balance.__init__(self, targetFlow, targetCompound, cfgPath)

    SinglePass.FractionalConversion(self)
    self.batchFlows = self.batchFlows.drop(
      columns=["Stoch Coeff", "Mass Flow (mtpd/20-min-batch)"])
    
    print(self.batchFlows)
    
    self.cols = {
          "mw": 'Molecular Weight (grams/mol)',
          "n": "Component Flow (kmol/20-min-batch)",
          "m": "Mass Flow (kg/20-min-batch)"
        }

  def IterFlows(self, write:bool=False, itr:int=0):
    cfg = self.c
    for stage in cfg["Stages"]:
      for unit in cfg["Stages"][stage]:
        print(f"stage/unit: {stage}/{unit}")
        unit_type = unit.split('-')[0]

        uo = cfg["Units"][unit]
        uo.setdefault("flow", {}).setdefault("reagents", {})
        uo.setdefault("source", {})
        
        conv = min(float(uo["conversion"]), 1) if "conversion" in uo else 1

        # 0 stage
        if stage == str(0):  

          if unit_type == "PSA": uops = uo["seperation"]["reagents"]
          else: uops = list(uo["reaction"]["reagents"].keys())

          # Set flows from material balance for initCompounds
          for comp in [x for x in uops if x in cfg["initCompounds"]]:
            
            row = self.batchFlows.loc[
              (self.batchFlows["Component"] == comp) & 
              (self.batchFlows[self.cols["m"]] < 0)
              ]
            
            uo["flow"]["reagents"][comp] = {
                    self.cols["n"]: row[self.cols["n"]].values[0],
                    self.cols["m"]: row[self.cols["m"]].values[0],
                  }

        # n stages               
        else:
          if unit_type != "PSA": p = "reaction"
          else: p = "seperation"

          # Get reagent flows from input Unit Operations
          for comp in uo[p]["reagents"]:
            for input in uo["inputs"]:
              inputuo = cfg["Units"][input]
              if comp in inputuo["flow"]["products"].keys():
                
                n = inputuo["flow"]["products"][comp][self.cols["n"]]
                m = inputuo["flow"]["products"][comp][self.cols["m"]]
                
                uo["flow"]["reagents"][comp] = {
                  self.cols["n"]: n,
                  self.cols["m"]: m
                }
            
            # Get reagent flows for sources (water/air)
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

        if unit_type != "PSA":
          if uo["flow"]["reagents"].keys() == uo["reaction"]["reagents"].keys():
            
            # Update reagents with recycles from previous state
            if "recycle" in uo.keys() and itr>0:
              for iuo in uo["recycle"].keys():

                recy = uo["recycle"][iuo]
                for comp in recy.keys():

                  with open(f'states/iter_{itr-1}.json', "r") as f: 
                    re = json.load(f)["Units"][iuo]["flow"]

                  for stream in ["products", "side"]:
                    if comp in re.get(stream, {}).keys():

                      n = re[stream][comp][self.cols["n"]]
                      m = re[stream][comp][self.cols["m"]]
                      
                      row = uo["flow"]["reagents"].get(comp, {})
                      
                      uo["flow"]["reagents"][comp] = {
                          self.cols["n"]: n + row.get(self.cols["n"], 0),
                          self.cols["m"]: m + row.get(self.cols["m"], 0)
                        }

            if len(uo["reaction"]["reagents"].keys()) == 1: 
              reagent = next(iter(uo["reaction"]["reagents"]))
            else: reagent = uo["limiting_reagent"]

            # Generate product flows
            for prod in uo["reaction"]["products"].keys():
              
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

            # Add unreacted reagents to product flows
            if conv < 1:
              for reagent in uo["reaction"]["reagents"].keys():
                rflow = uo["flow"]["reagents"][reagent]

                uo["flow"]["products"][reagent] = {
                  self.cols["n"]: rflow[self.cols["n"]] * (1 - conv),
                  self.cols["m"]: rflow[self.cols["m"]] * (1 - conv),
                }
            
          else:
            e = (set(uo["flow"]["reagents"].keys()) ^ 
                  set(uo["reaction"]["reagents"].keys()))
            
            raise Exception(f"reaction reagents not found for {unit}, missing: {e}")
        
        elif unit_type == "PSA":
          output = uo["seperation"]["side"] + uo["seperation"]["products"]
          if set(list(uo["flow"]["reagents"].keys())) == set(output):
            for reagent in uo["seperation"]["reagents"]:
              if reagent in uo["seperation"]["products"]: eta, prod = conv, "products"
              else: eta, prod = 1, "side"

              if "_" in reagent: sub = reagent.split('_')[0]
              else: sub = reagent
              
              n = (uo["flow"]["reagents"][reagent][self.cols["n"]]
                   * eta).__abs__()
              
              m = (self.q(n, 'kmol/batch') * 
                    self.q(self.subs[sub].mass, 'g/mol')
                    ).to("kg/batch")
              
              flow_data = {
                reagent: {
                    self.cols["n"]: n,
                    self.cols["m"]: m.magnitude,
                }}

              if prod in uo["flow"]: uo["flow"][prod].update(flow_data)
              else: uo["flow"][prod] = flow_data
            
            if conv < 1:
              for reagent in uo["seperation"]["products"]:
                rflow = uo["flow"]["seperation"][reagent]

                uo["flow"].setdefault(prod, {}).update({
                  prod: {
                    reagent: {
                      self.cols["n"]: rflow[self.cols["n"]] * (1 - conv),
                      self.cols["m"]: rflow[self.cols["m"]] * (1 - conv),
                  }}})
          else:
            e = (set(uo["flow"]["reagents"].keys()) ^ 
                  set(uo["seperation"]["reagents"]))
            print(e)
            
            raise Exception(f"Seperation reagents not found for {unit}, missing: {e}")
          
        else: raise Exception(f"Missing reagents {stage} {unit}")

        pprint.pprint(f"=== Stage: {stage}, Unit: {unit} ===")
        #pprint.pprint(self.c["Units"][unit])
      
      if stage == str(4): break
    
    if write:
      with open(f'states/iter_{itr}.json', 'w') as js:
        json.dump(self.c, js, indent=4)

  def features(self):
    SinglePass.IterFlows(self, False)

    SinglePass.ThermalProperties(self, True)
    
if __name__ == "__main__":
  core, feat = True, False
  x = SinglePass()

  if core: x.IterFlows(True)
  if feat: x.features()
  