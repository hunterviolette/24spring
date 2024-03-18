import os
import pandas as pd
import json
import numpy as np

from typing import List
from thermo import Chemical, Mixture

if os.getcwd().split("/")[-1].endswith("src"):
  from unit_registry import UnitConversion
else: 
  from src.unit_registry import UnitConversion

class Therm(UnitConversion):

  def __init__(self, cfgPath: str = "./cfg.json") -> None:
    super().__init__()

    with open(cfgPath, "r") as f: self.c = json.load(f)

  def __thermdata__(self, dir: str = "./data"):
    for f in [x for x in os.listdir(dir)
              if os.path.isfile(f"{dir}/{x}")
              and x.endswith(".csv")]:
      
      setattr(self, f.split('.')[0], pd.read_csv(f"{dir}/{f}"))
  
  @staticmethod
  def DeltaMixProperty( 
              t1: int = 900, # output temepratrue (K)
              t0: int = 1000, # input temperature (K) 
              p1: int = 1E5, # output pressure (Pa)  
              p0: int = 1E5, # output pressure (Pa) 
              chems: List[str] = ["ammonia", "7647-01-0"], 
              zs: List[float] = [.63, .34],
              prop: str = 'Hm'# Um 
            ):
    """
    Calculate the change in a thermodynamic property of a mixture between two states
    based on the property name given as a string.
    
    Parameters:
    - t1: Output temperature in Kelvin.
    - t0: Input temperature in Kelvin.
    - p1: Output pressure in Pascal.
    - p0: Input pressure in Pascal.
    - chems: List of chemical identifiers.
    - zs: Mole fractions of the chemicals.
    - prop: Property name of the thermodynamic property ('Hm' for enthalpy, 'Um' for internal energy).
    
    if prop == Hm # Simulate Q in heatX
      Returns: Change in molar enthalpy of mixture between two states in J/mol

    if prop == Um # Simulate W in pump
      Returns: Change in molar internal energy of mixture between two states in J/mol
    """
    return ( # J/mol
        Mixture(chems, zs=zs, T=t1, P=p1).__getattribute__(prop) -
        Mixture(chems, zs=zs, T=t0, P=p0).__getattribute__(prop)
      )

  def CheckChemical(self, name):
    try:
      ch = Chemical(name)
      return True
    except Exception as e:
      print(f"Data for {name} is not available. Error: {e}")
      return False
        
  def ThermalProperties(self, write:bool=False, itr:int=0):
    cfg = self.c

    # Overall properties
    for chem in [[key, value] for key, value in 
                cfg["Compounds"].items()]:
      try:
        c = Chemical(chem[1])
        cfg.setdefault("Chemical Properties", {}).update({
        chem[0]: {
          "CAS": chem[1],
          "Tb (K)": c.Tb,
          "Tc (K)": c.Tc,
          "Tm (K)": c.Tm,
          "MW (g/mol)": c.MW,
          "T_flash (K)": c.Tflash,
          "T_flash_source": c.Tflash_source
        }})
      except Exception as e:
        cfg.setdefault("Chemical Properties", {}).update({
          chem[0]: {"No properties found": str(e)}})

    # Unit properties
    chms = [key for key, value in 
            cfg["Compounds"].items() 
            if Therm.CheckChemical(self, value)]

    chemProps = [key for key, value 
                in cfg["Compounds"].items() if key in chms]
        
    ## Unit properties
    for unit in cfg["Units"]:
      uo = cfg["Units"][unit]
      
      if not "PSA" in unit: 
        rxn = uo["reaction"]
        for reag in rxn["reagents"].keys():
          
          for iop in uo.get("inputs", {}):
            if "PSA" in iop: 
              reags = cfg["Units"][iop]["seperation"]["products"]
            else: 
              reags = list(cfg["Units"][iop]["reaction"]["products"].keys())
            
            if reag in reags:
              t = cfg["Units"][iop]["temperature"].split(" ")
              t0 = self.q(float(t[0]), t[1]).to("K")
              
              p = cfg["Units"][iop]["pressure"].split(" ")
              p0 = self.q(float(p[0]), p[1]).to("Pa")
          
            print(reag, chemProps)
            if reag in chemProps:
              t = uo["temperature"].split(" ")
              t1 = self.q(float(t[0]), t[1]).to("K")
              
              p1 = uo["pressure"].split(" ")
              p1 = self.q(float(p[0]), p[1]).to("Pa")

              chem = Chemical(
                      cfg["Compounds"][reag],
                      T=t0.magnitude,
                      P=p0.magnitude
                      )
              
              try:
                h = self.q(chem.calc_H(
                            T=t1.magnitude, 
                            P=p1.magnitude
                          ), 'J/mol')
                              
                eh = self.q(chem.calc_H_excess(
                            T=t1.magnitude, 
                            P=p1.magnitude
                          ), 'J/mol')
                
                s = self.q(chem.calc_S(
                            T=t1.magnitude, 
                            P=p1.magnitude
                          ), 'J/mol/K')
                
                es = self.q(chem.calc_S_excess(
                            T=t1.magnitude, 
                            P=p1.magnitude
                          ), 'J/mol/K')
              except: 
                h = self.q(0, 'kJ/mol')
                eh = self.q(0, 'kJ/mol')
                s = self.q(0, 'kJ/mol/K')
                es = self.q(0, 'kJ/mol/K')

              uo.setdefault("properties", {}).update({
                reag : {
                  "t0 (K)": t0.to("K").magnitude,
                  "t1 (K)": t1.to("K").magnitude,
                  "p0 (bar)": p0.to("bar").magnitude,
                  "p1 (bar)": p1.to("bar").magnitude,
                  "dH (kJ/mol)": h.to("kJ/mol").magnitude,
                  "excess dH (kJ/mol)": eh.to("kJ/mol").magnitude,
                  "dS (kJ/mol/K)": s.to("kJ/mol/K").magnitude,
                  "excess dS (kJ/mol/K)": es.to("kJ/mol/K").magnitude,
                }})
                      
              #print(t0, t1, p0, p1, h)
              
            else:
              uo.setdefault("properties", {}).update({
                reag : "No properties found"})

    if write:
      with open(f'states/iter_{itr}.json', 'w') as j:
        json.dump(cfg, j, indent=4)

if __name__ == "__main__":
  x = Therm().ThermalProperties()
  