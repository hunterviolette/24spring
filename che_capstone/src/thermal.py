import os
import pandas as pd
import json
import numpy as np

from typing import List
from thermo import Chemical, Mixture, volume

if os.getcwd().split("/")[-1].endswith("src"):
  from unit_registry import UnitConversion
else: 
  from src.unit_registry import UnitConversion

class Therm(UnitConversion):

  def __init__(self, cfgPath: str = "./iter_7.json") -> None:
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
              chem1: List[str] = ["ammonia", "7647-01-0"], 
              chem0: List[str] = ["ammonia", "7647-01-0"], 
              z0: List[float] = [.63, .34],
              z1: List[float] = [.63, .34],
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
        Mixture(chem1, zs=z1, T=t1, P=p1).__getattribute__(prop) -
        Mixture(chem0, zs=z0, T=t0, P=p0).__getattribute__(prop)
      )

  @staticmethod
  def CheckChemical(name):
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
            if Therm.CheckChemical(value)]

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

  @staticmethod
  def ReactionDict(uo: dict, cfg: dict, col: str = "Flow (kmol/batch)"):
    rxn = {}
    for side in ["reagents", "products"]:
      comps, molFracs, name = [], [], []
      for reagent in uo["reaction"][side].keys():
        inDB = Therm.CheckChemical(cfg["Compounds"][reagent])
        if inDB:
          comps.append(cfg["Compounds"][reagent])
          
          molFracs.append(uo["flow"][side][reagent][col] / 
                            sum(uo["flow"][side][key][col] 
                                for key in uo["flow"][side].keys()
                                if key in uo["reaction"][side]
                                and inDB
                              ))
          name.append(reagent)
          
        else: rxn.setdefault("Missing data", {}).update({reagent: side})
      
      rxn[side] = {
              "Compounds": comps, 
              "Mole Fractions": molFracs,
              "Name": name
            }
      
    return rxn

  def dH_Mixture(self):
    cfg = self.c

    for unit in [x for x in cfg["Units"].keys()
                if x.split("-")[0] in ["R"]]:
      
      uo = cfg["Units"][unit]
      rxn = Therm.ReactionDict(uo, cfg)
      
      temp, pres = uo["temperature"].split(" "), uo["pressure"].split(" ")

      t = self.q(float(temp[0]), temp[1]).to("degK").magnitude
      p = self.q(float(pres[0]), pres[1]).to("Pa").magnitude

      if unit == "R-102" and "Mg3N2" in uo["reaction"]["products"].keys():
        # 3 Mg + N2 -> Mg3N2
        cp = self.q(np.interp(1000, [300, 700], [.1046, .1168]), 'kJ/(mol*degK)')
        prods = (cp * self.q(100, 'degK')).to('kJ/mol')

        reags = self.q(Mixture(
                          IDs=rxn["reagents"]["Compounds"], 
                          zs=rxn["reagents"]["Mole Fractions"],
                          T=t, P=p
                        ).__getattribute__("Hm"),'J/mol').to('kJ/mol')
        
        dH = (prods - reags).magnitude

      elif unit == "R-103" and "Mg3N2" in uo["reaction"]["reagents"].keys():
        # Mg3N2 + 6 HCl -> 3 MgCl2 + 2 NH3
        cp = self.q(np.interp(1100, [300, 700], [.1046, .1168]), 'kJ/(mol*degK)')
        mg3n2 = (cp * self.q(-100, 'degK')).to('kJ/mol')

        reags = self.q(Mixture(
                          IDs=rxn["reagents"]["Compounds"], 
                          zs=rxn["reagents"]["Mole Fractions"],
                          T=t, P=p
                        ).__getattribute__("Hm"),'J/mol').to('kJ/mol') + mg3n2
        
        prods = self.q(Mixture(
                          IDs=rxn["products"]["Compounds"], 
                          zs=rxn["products"]["Mole Fractions"],
                          T=t, P=p
                        ).__getattribute__("Hm"),'J/mol').to('kJ/mol')
        
        dH = (prods - reags).magnitude
        
      elif not any(rxn["reagents"]["Compounds"]) or not any(rxn["reagents"]["Mole Fractions"]) or \
          not any(rxn["products"]["Compounds"]) or not any(rxn["products"]["Mole Fractions"]):
            
            dH = "Missing property data"
      else:
        dH = self.q(Therm.DeltaMixProperty(
                      t1=t,
                      t0=t,
                      p1=p,
                      p0=p,
                      chem1=rxn["products"]["Compounds"],
                      chem0=rxn["reagents"]["Compounds"],
                      z1=rxn["products"]["Mole Fractions"],
                      z0=rxn["reagents"]["Mole Fractions"],
                    ), "J/mol").to("kJ/mol").magnitude
        
      uo["reaction"]["dH mixtures (kJ/mol)"] = {
            "overall": dH
        }

  def HeatRxn(self):
    Therm.__thermdata__(self, './src/data')
    cfg = self.c

    for unit in [x for x in cfg["Units"].keys()
                if x.split("-")[0] in ["R"]]:
      
      uo = cfg["Units"][unit]
      rxn = Therm.ReactionDict(uo, cfg)
      
      temp, pres = uo["temperature"].split(" "), uo["pressure"].split(" ")

      t = self.q(float(temp[0]), temp[1]).to("degK").magnitude
      p = self.q(float(pres[0]), pres[1]).to("Pa").magnitude

      prods, reags = 0, 0
      for side in ["products", "reagents"]:
        for i, comp in enumerate(rxn[side]["Compounds"]):
          hfm = self.q(Chemical(ID=comp, T=t, P=p
                            ).Hfm, 'J/mol').to('kJ/mol')
          
          stoich = abs(uo["reaction"][side][rxn[side]["Name"][i]])

          if side == "reagents": reags += hfm * stoich
          else: prods += hfm * stoich

      for comp in rxn.setdefault("Missing data", {}).keys():
        stoich = abs(uo["reaction"][rxn["Missing data"][comp]][comp])
        
        hfm = self.q(Therm.EnthalpyFormation(
          self.formations.loc[self.formations["Component"] == comp],
          t
        ), 'kJ/mol')
          
        if rxn["Missing data"][comp] == "products": prods += hfm * stoich
        else: reags += hfm * stoich

      if not isinstance(prods, int) and not isinstance(reags, int):

        heatRxn = (prods - reags).to("kJ/mol")
        uo["reaction"]["heat of reaction (kJ/mol)"] = {
                            "overall": heatRxn.magnitude,
                            "products": prods.magnitude,
                            "reagents": reags.magnitude
                          }
      else: uo["reaction"]["heat of reaction (kJ/mol)"] = {
                            "Property Data": "Missing"
                          }
  
  def Reactor_Q(self):
    Therm.dH_Mixture(self)
    Therm.HeatRxn(self)
    
    cfg = self.c
    for unit in [x for x in cfg["Units"].keys()
                if x.split("-")[0] in ["R"]]:
      
      uo = cfg["Units"][unit]

      q_mixture = uo["reaction"]["dH mixtures (kJ/mol)"]["overall"]
      q_rxn = uo["reaction"]["heat of reaction (kJ/mol)"]["overall"]
      q_overall = q_mixture + q_rxn

      reagent = Therm.LimitingReagent(uo)
      n = (uo["flow"]["reagents"][reagent][self.cols["n"]]).__abs__()

      q_dot = self.q(q_overall, 'kJ/mol') * self.q(n, 'kmol/batch').to_base_units()

      uo["reaction"]["Overall Molar Heat (kJ/mol)"] = q_overall
      uo["reaction"]["Limiting Reagent Flow (kmol/batch)"] = n
      uo["reaction"]["Utility (kW)"] = q_dot.magnitude
      
  def Electrolysis_Q(self):
    cfg = self.c

    for unit in [x for x in cfg["Units"].keys()
                if x.split("-")[0] in ["EL"]]:
      
      uo = cfg["Units"][unit]
      
      reagent = Therm.LimitingReagent(uo)
      n = uo["flow"]["reagents"][reagent][self.cols["n"]]

      uo["reaction"]["Utility (kW)"] = (

        self.q(n, 'kmol/batch') 
        * 2 * self.q(1,'faraday_constant')
        * self.q(3,'V')

      ).to('kW').magnitude

  @staticmethod
  def EnthalpyFormation(d, t):
    '''
    Only for solids, used for Mg3N2
    '''
    return (
      d["A"].values[0] + 
      d["B"].values[0] * t + 
      d["C"].values[0] * t**2 + 
      d["D"].values[0] * t**3 +
      d["E"].values[0] * t**4
    )

  def init_EL101_Q(self):
    chem = "7786-30-3"
    h0 = Chemical(
              ID=chem, 
              T=self.q(25, 'degC').to('degK').magnitude,
              P=self.q(1, 'atm').to('Pa').magnitude
            ).Hm
    
    h1 = Chemical(
              ID=chem, 
              T=self.q(1000, 'degK').magnitude,
              P=self.q(1, 'atm').to('Pa').magnitude
            ).Hm
    
    dH = self.q(h1 - h0, 'J/mol').to('kJ/mol')
    n = self.q(2.2019900907798, 'kmol')

    q = (dH * n / self.q(20, 'min')).to('kW')

    print(f"Initial MgCl2 Q (kJ/mol): {dH}",
          f"Power (kW): {q}", sep='\n',)

  def R102_pQ(self):
    chem = "magnesium"
    h0 = Chemical(
              ID=chem, 
              T=self.q(1000, 'degK').magnitude,
              P=self.q(1, 'atm').to('Pa').magnitude
            ).Hm
    
    h1 = Chemical(
              ID=chem, 
              T=self.q(1100, 'degK').magnitude,
              P=self.q(1, 'atm').to('Pa').magnitude
            ).Hm
    
    dH_mg = self.q(h1 - h0, 'J/mol').to('kJ/mol')

    chem = "nitrogen"
    h0 = Chemical(
              ID=chem, 
              T=self.q(25, 'degC').to('degK').magnitude,
              P=self.q(1, 'atm').to('Pa').magnitude
            ).Hm
    
    h1 = Chemical(
              ID=chem, 
              T=self.q(1100, 'degK').magnitude,
              P=self.q(1, 'atm').to('Pa').magnitude
            ).Hm
    
    dH_n2 = self.q(h1 - h0, 'J/mol').to('kJ/mol')
    
    print(
        f"Magnesium dH (kJ/mol): {dH_mg}",
        f"Nitrogen dH (kJ/mol): {dH_n2}",
        sep='\n'
      )

  def AmmoniaVolume(self):
    for y in [225, 250, 273, 298]:
      t = self.q(y,'degK').magnitude

      print(f"{y} Kelvin")
      for x in [1, 5, 10, 15, 20, 30, 50]:
        p = self.q(x,'bar').to("Pa").magnitude

        c = Chemical("ammonia", T=t, P=p)
        
        Vs = self.q(
                volume.COSTALD(
                  T=t, Tc=c.Tc, 
                  Vc=c.Vc, omega=c.omega, 
                ), 'm**3/mol')

        rho = self.q(
                volume.COSTALD_compressed(
                  T=t, P=p, Psat=c.Psat,
                  Tc=c.Tc, Pc=c.Pc, omega=c.omega, 
                  Vs=Vs.magnitude
                )
                , 'm**3/mol')

        print(f"phase: {c.phase} pressure: {x} bar, {rho}")

if __name__ == "__main__":
  x = Therm()
  #x.init_EL101_Q()
  #x.R102_pQ()
  x.Ammonia()