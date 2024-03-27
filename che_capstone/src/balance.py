import pandas as pd
import pint
import json 
import os

from chempy import Substance

if os.getcwd().split("/")[-1].endswith("src"):
  from unit_registry import UnitConversion
else: 
  from src.unit_registry import UnitConversion

class Air: 
  '''
  Very basic implementation to mimic ChemPy.mass
  '''
  mass = 28.9647
  n2frac = .7808
  o2frac = .20947
  arfrac = .00934
  co2frac = .00035


  CompList = ["O2", "Ar", "CO2"]

  @staticmethod
  def AirFrac():
    return {
        "N2": Air.n2frac,
        "O2_air": Air.o2frac,
        "Ar_air": Air.arfrac,
        "CO2_air": Air.co2frac
      }

  @staticmethod
  def MassPercent(comp: str = "N2", retfrac:bool=False):
    if comp in ["N2", "O2", "Ar", "CO2"]:
      uReg = pint.UnitRegistry(autoconvert_offset_to_baseunit = True)
      q = uReg.Quantity
      
      fracs = {
        "N2": Air.n2frac,
        "O2": Air.o2frac,
        "Ar": Air.arfrac,
        "CO2": Air.co2frac
      }
      if retfrac: return fracs[comp]

      return (
              q(Substance.from_formula(comp).mass, "g/mol")
              / q(Air.mass, 'g/mol') * fracs[comp]
            ).magnitude

class Balance(UnitConversion): 
  def __init__(self, 
              cfgPath: str = "../cfg.json"
            ) -> None:
    
    UnitConversion.__init__(self)
    
    with open(cfgPath, "r") as f: self.c = json.load(f)

    self.subs = {
      x: Substance.from_formula(x) for x in self.c["Compounds"]}
    
    self.subs["Air"] = Air

  def MaterialBalance(self):
    cfg = self.c

    tf = cfg["Basis"]["Target Flow"].split(" ")

    targetCompound = cfg["Basis"]["Target Compound"]
    targetFlow = self.q(float(tf[0]), tf[1])
    targetMW = self.q(self.subs[targetCompound].mass, 'gram/mol')
    targetStoich = float(cfg["Basis"]["Overall Reaction"][targetCompound]["stoich"])

    self.bph = cfg.setdefault("Basis", {}).setdefault("Batches per Hour", 1)

    self.ureg.define(f'batch = {60 / self.bph} * min')
    self.q = self.ureg.Quantity

    molph = (targetFlow / targetMW).to(f'kmol/batch')

    df = pd.DataFrame()
    for comp, v in self.c["Basis"]["Overall Reaction"].items():
      mw = self.q(self.subs[comp].mass, 'gram/mol')

      molFlow = (molph / targetStoich * v["stoich"]).__round__(3)
      massFlow = (molFlow * mw).to("mtpd").__round__(3)

      df = pd.concat([df, 
                      pd.DataFrame({
                        "Component": [comp],
                        "Stoch Coeff": [v["stoich"]],
                        "Molecular Weight (grams/mol)": [mw.magnitude],
                        f"Component Flow (kmol/h)": [molFlow.to('kmol/h').magnitude],
                        "Mass Flow (mtpd)": [massFlow.magnitude],
                        "Mass Flow (kg/h)": [massFlow.to('kg/h').magnitude],
                        f"Flow (kmol/batch)": [molFlow.to('kmol/batch').magnitude],
                        f"Flow (kg/batch)": [massFlow.to('kg/batch').magnitude]
                      })
                    ])

    print(df)
    self.mb, self.targetCompound, self.targetFlow = df, targetCompound, targetFlow

  def ClosedBalance(self):
    pass

if __name__ == "__main__":
  #y = Air.MassPercent("N2")
  #print(y)

  x = Balance()

  x.MaterialBalance()
