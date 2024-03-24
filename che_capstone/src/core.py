import pandas as pd
import pprint
import json
import os

from typing import Optional

if os.getcwd().split("/")[-1].endswith("src"):
  from balance import Balance, Air
  from thermal import Therm
else: 
  from src.balance import Balance, Air
  from src.thermal import Therm

class SinglePass(Balance, Therm):

  def __init__(self, 
              cfgPath: str = "../cfg.json"
            ) -> None:
    
    self.cPath = cfgPath
    Balance.__init__(self, cfgPath)

    SinglePass.MaterialBalance(self)

    print(self.mb)
    
    self.cols = {
          "mw": 'Molecular Weight (grams/mol)',
          "n": "Flow (kmol/batch)",
          "m": "Flow (kg/batch)"
        }
    
  def __cfg__(self):
    # Need to open fresh config or previous state carries over
    with open(self.cPath, "r") as f: self.c = json.load(f)

  def IterFlows(self, write:bool=False, itr:int=0, excess:float=1):
    SinglePass.__cfg__(self)
    cfg = self.c

    if itr >0: 
      with open(f'states/iter_{itr-1}.json', "r") as f: 
        pstate = json.load(f)

    print('======', f'Iteration: {itr}', '======', sep='\n')
    for unit, uo in cfg["Units"].items():
        flow = uo.setdefault("flow", {})
        flow.setdefault("reagents", {})
        flow.setdefault("products", {})
        flow.setdefault("side", {})
        uo.setdefault("source", {})
        uo.setdefault("depends_on", {})

    # Order of how unit operations are called
    for stage in cfg["Stages"]:
      # For each unit in each stage 
      for unit in cfg["Stages"][stage]:
        pprint.pprint(f"=== Init Stage: {stage}, Unit: {unit}, Iteration: {itr} ===")

        uo = cfg["Units"][unit]
        conv = min(float(uo["conversion"]), 1) if "conversion" in uo else 1

        # variable flow rate dependent sources  
        for k,v in uo["depends_on"].items():
          print("RUNNING DEP HERE _______ ", k)

          dcomp = v["dependent compound"]

          state_dict = pstate["Units"] if v["state"] <0 and itr>0 else cfg["Units"]

          dn = state_dict.get(k, {}).get("flow", {}).get("products", {}
                        ).get(dcomp, {}).get(self.cols["n"], 0)

        
          comp, stoich = v["compound"], v["stoich"]
          cf = uo["flow"]["reagents"].get(comp, {}).get(self.cols["n"], 0)
          n = dn * stoich + cf
          print(n, dn, cf)

          m = (self.q(n, 'kmol/batch') * 
                      self.q(self.subs[comp].mass, 'g/mol')
                    ).to("kg/batch")

          print(" ".join([
                  f"getting dependent flows from {k} from {dcomp},",
                  f"got {dn:.2f} {self.cols['n']} with stoich {stoich},",
                  f"{comp} is {n:.2f} {self.cols['n']}"
                ]))
          
          if comp == "N2":
            n /= Air.n2frac

            airFracs = Air.AirFrac()
            for comp in airFracs.keys():

              uo["flow"]["reagents"][comp] = {
                    self.cols["n"]: (n) * airFracs[comp],
                  }
          else:
            uo["flow"]["reagents"][comp] = {
                    self.cols["n"]: n,
                    self.cols["m"]: m.magnitude,
                  }

        # 0 stage, get initial reagent flows on Basis Compound/flow   
        if stage == str(0) and itr == 0:  
          
          if "PSA" in unit: uops = uo["seperation"]["reagents"]
          else: uops = list(uo["reaction"]["reagents"].keys())

          # Set flows from material balance
          for comp in [x for x in uops if x in cfg["Basis"]["Get Flows"]]:
            
            row = self.mb.loc[
              (self.mb["Component"] == comp) & 
              (self.mb[self.cols["m"]] < 0)
              ] 

            n = abs(float(row[self.cols["n"]].values[0])) * excess
            m = abs(float(row[self.cols["m"]].values[0])) * excess
            
            print(" ".join([
                f"getting flows from OMB for {comp},",
                f"got {n:.2f} {self.cols['n']}",
              ]))
            
            uo["flow"]["reagents"][comp] = {
                    self.cols["n"]: n,
                    self.cols["m"]: m,
                  }

        # stage>0 and itr>0 stages, get reagent flows from input/recycle unit               
        else:
          if not "PSA" in unit: p = "reaction"
          else: p = "seperation"

          # Get reagent flows from input Unit Operations
          for comp in uo[p]["reagents"]:
            for input in uo.get("inputs", {}):
              inputuo = cfg["Units"][input]
              if comp in inputuo["flow"]["products"].keys():
                
                n = inputuo["flow"]["products"][comp][self.cols["n"]]
                m = inputuo["flow"]["products"][comp][self.cols["m"]]

                print(" ".join([
                    f"getting flows from {input} for {comp},",
                    f"got {n:.2f} {self.cols['n']}",
                  ]))
                    
                uo["flow"]["reagents"][comp] = {
                  self.cols["n"]: n,
                  self.cols["m"]: m
                }

          # Update reagents with recycles from previous state
          if "recycle" in uo.keys() and itr>0:
            for iuo in uo["recycle"].keys():

              recy = uo["recycle"][iuo]
              for comp in recy.keys():

                for stream in ["products", "side"]:
                  if comp in pstate["Units"][iuo]["flow"][stream].keys():
                    vv = pstate["Units"][iuo]["flow"][stream][comp]
                    
                    n, m = vv[self.cols["n"]], vv[self.cols["m"]]

                    print(" ".join([
                      f"getting recycle flows from {iuo} for {comp},",
                      f"from {stream} got {n:.2f} {self.cols['n']}",
                    ]))
                    
                    currentFlow = uo["flow"]["reagents"].get(comp, {}).get(self.cols["n"], 0)
                    if currentFlow > 0:
                      print(f'New reagent flow for {comp}: {n+currentFlow:.2f}')
                    
                    uo["flow"]["reagents"][comp] = {
                        self.cols["n"]: n + currentFlow,
                        self.cols["m"]: m + currentFlow
                      }
                            
        # Do reaction
        if not "PSA" in unit:
          if uo["flow"]["reagents"].keys() == uo["reaction"]["reagents"].keys():

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

              uo["flow"]["products"][prod] = {
                    self.cols["n"]: n,
                    self.cols["m"]: m.magnitude,
                  }

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
        
        # Do seperation
        elif "PSA" in unit:
          output = uo["seperation"]["side"] + uo["seperation"]["products"]
          if set(list(uo["flow"]["reagents"].keys())) == set(output):
            # Generate product/side flows
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
            
            #If seperation isnt 100%
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

        pprint.pprint(f"=== Exc Stage: {stage}, Unit: {unit}, Iteration: {itr} ===")
          
    if write:
      with open(f'states/iter_{itr}.json', 'w') as js:
        json.dump(self.c, js, indent=4)

  def FlowFeatures(self, itr):
    SinglePass.dH_Mixture(self)
    SinglePass.HeatRxn(self)

    with open(f'states/iter_{itr}.json', 'w') as js:
        json.dump(self.c, js, indent=4)
    
if __name__ == "__main__":
  core, feat = True, False
  x = SinglePass()

  if core: x.IterFlows(True)
  if feat: x.FlowFeatures()
  