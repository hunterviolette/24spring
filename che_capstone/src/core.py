import pandas as pd
import pprint
import json

if __name__ == "__main__":
  from v3_balance import Balance
  from thermal_analysis import Therm
else:
  from src.v3_balance import Balance
  from src.thermal_analysis import Therm

class SinglePass(Balance, Therm):

  def __init__(self, 
              targetFlow: int = 1, # mtpd
              targetCompound: str = "NH3"  
            ) -> None:
    
    Balance.__init__(self, targetFlow, targetCompound)

    SinglePass.FractionalConversion(self)
    self.batchFlows = self.batchFlows.drop(
      columns=["Stoch Coeff", "Mass Flow (mtpd/20-min-batch)"])
    
    print(self.batchFlows)
    
    self.cols = {
          "mw": 'Molecular Weight (grams/mol)',
          "n": "Component Flow (kmol/20-min-batch)",
          "m": "Mass Flow (kg/20-min-batch)"
        }

  def core(self, write:bool=False):
    cfg = self.c
    for stage in cfg["Stages"]:
      for unit in cfg["Stages"][stage]:
        print(f"stage/unit: {stage}/{unit}")
        unit_type = unit.split('-')[0]

        uo = cfg["Units"][unit]
        uo.setdefault("flow", {}).setdefault("reagents", {})
        uo.setdefault("source", {})
        
        conv = min(float(uo["conversion"]), 1) if "conversion" in uo else 1
        if stage == str(0): # init stage 

          if unit_type == "PSA": uops = uo["seperation"]["reagents"]
          else: uops = list(uo["reaction"]["reagents"].keys())

          for comp in [x for x in uops if x in cfg["initCompounds"]]:
            
            row = self.batchFlows.loc[
              (self.batchFlows["Component"] == comp) & 
              (self.batchFlows[self.cols["m"]] < 0)
              ]
            
            uo["flow"]["reagents"][comp] = {
                    self.cols["n"]: row[self.cols["n"]].values[0],
                    self.cols["m"]: row[self.cols["m"]].values[0],
                  }
                          
        else: # n stages
          if unit_type != "PSA": p = "reaction"
          else: p = "seperation"

          for comp in uo[p]["reagents"]:
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

        if unit_type != "PSA":
          if uo["flow"]["reagents"].keys() == uo["reaction"]["reagents"].keys():
          
            if len(uo["reaction"]["reagents"].keys()) == 1: 
              reagent = next(iter(uo["reaction"]["reagents"]))
            else: reagent = uo["limiting_reagent"]

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
        pprint.pprint(self.c["Units"][unit])
      
      if stage == str(4): break
    
    if write: 
      with open('states/iter_0.json', 'w') as json_file:
        json.dump(self.c, json_file, indent=4)

  def features(self):
    SinglePass.core(self)

if __name__ == "__main__":
  x = SinglePass()
  x.core()