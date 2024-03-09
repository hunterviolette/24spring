import periodictable
import pandas as pd
import pint
import json 

from chempy import Substance

'''
# Electrolysis
  MgCl2 -> Mg + Cl2

N2 Source:
  3Mg + N2 -> Mg3 + N2 -> Mg3N2

H2 Source: 
  3Cl2 + 3H2O -> 6HCl + 1.5O2

Main rxn:
  Mg3N2 + 6HCl = 3MgCl2 + 2NH3

Overall:
  3MgCl2 + N2 + 3H2O -> 2NH3 + 1.5O2

'''

class Air: 
  mass = 28.9647
  n2frac = .7808

class Balance: 
  def __init__(self, targetflow: int = 1 # in metric ton per day 
            ) -> None:
    
    uReg = pint.UnitRegistry(autoconvert_offset_to_baseunit = True)
    uReg.define('mtpd = metric_ton / day')
    uReg.define('batch = 1 / (20*min)')
    uReg.define('million_BTU = 1e6 * BTU')
    uReg.define('dollar = [currency] = $')

    self.q = uReg.Quantity
    self.targetFlow = self.q(targetflow, 'mtpd')

    with open("./cfg.json", "r") as f: self.c = json.load(f)

    self.subs = {
      x: Substance.from_formula(x) for x in self.c["Compounds"]}
    
    self.subs["Air"] = Air

  def OverallMaterialBalance(self, verbose:bool=False) -> None:

    targetMW = self.q(self.subs["NH3"].mass, 'gram/mol')
    molph = (self.targetFlow / targetMW).to('kmol/h')
    
    # rxn stoich
    ## 3 MgCl2 + N2 + 3 H2O  + 6 HCl -> 2 NH3 + 1.5 O2 + 3 MgCl2
    
    df = pd.DataFrame()
    for x in [[-3, "MgCl2"], [-1, "N2"], [-3, "H2O"],
              [2, "NH3"], [1.5, "O2"], [3, "MgCl2"],
              [-1, "Mg3N2"], [-6, "HCl"]]:
      
      mw = self.q(self.subs[x[1]].mass, 'gram/mol')
      stoich = x[0]

      molFlow = (molph / 2 * stoich).__round__(3)
      massFlow = (molFlow * mw).to("mtpd").__round__(3)

      df = pd.concat([df,
                      pd.DataFrame({
                            "Component": [x[1]],
                            "Stoch Coeff": [stoich],
                            "Molecular Weight (grams/mol)": [mw.magnitude],
                            "Component Flow (kmol/h)": [molFlow.magnitude],
                            "Mass Flow (mtpd)": [massFlow.magnitude],
                            "Mass Flow (kg/h)": [massFlow.to('kg/h').magnitude]
                      })])
    
    if "N2" in df["Component"].unique():
      mw = self.subs["Air"].mass
      m = df.loc[df["Component"] == "N2"]["Mass Flow (mtpd)"].values[0] / Air.n2frac
      n = (self.q(m, "mtpd") / self.q(mw, 'g/mol')).to('kmol/h')
      
      df = pd.concat([df, 
                      pd.DataFrame({
                              "Component": ["Air"],
                              "Stoch Coeff": [Air.n2frac],
                              "Molecular Weight (grams/mol)": [mw],
                              "Component Flow (kmol/h)": [n.magnitude],
                              "Mass Flow (mtpd)": [m],
                              "Mass Flow (kg/h)": [self.q(m, "mtpd").to('kg/h').magnitude]
                        })])

    self.ombal = df.reset_index(drop=True).round(5).sort_values("Mass Flow (mtpd)")
    if verbose:
      print('=== ===',
            f"Target: {self.targetFlow} NH3 (stored)",
            f"NH3 metric ton per day: {(molph*targetMW).to('mtpd').__round__(4)}",
            '=== Overall Material Balance ===',
            self.ombal,
            '',
            sep='\n\n')
    
  def ReactorVolume(self):
    Balance.OverallMaterialBalance(self)
    col = "Mass Flow (mtpd)"

    molFlow = self.q(self.ombal.at[3, 'Component Flow (kmol/h)'], 'kmol/h')
    nh3syn = self.q(10**-6, 'mol/cm**2/s')

    nh3 = self.q(self.ombal.at[3, col], 'mtpd')
    hcl = self.q(self.ombal.at[7, col], "mtpd")
    mg3n2 = self.q(self.ombal.at[6, col], "mtpd")
    mgcl2 = self.q(self.ombal.at[5, col], "mtpd")

    print('=== R-104 ===', 
          f'Input = HCl (g): {hcl}, Mg3N2 (s): {mg3n2}',
          f'Output = MgCl2 (l): {mgcl2}, NH3 (g): {nh3}',
          f'(Output - Input) = {(nh3 + mgcl2 + hcl + mg3n2).__round__(4)}',
          f"NH3 (total/stored/recycled): {nh3}, {nh3*2/8}, {nh3*6/8}",
          '=== R-104 Design ===',
          f"NH3 Synthesis Rate: {nh3syn}",
          f"Area (net): {(molFlow / nh3syn).to('m**2').__round__(3)}",
          f"Area (total): {(molFlow / nh3syn * 4).to('m**2').__round__(3)}",
          '',
          sep='\n\n')

  def FractionalConversion(self, verbose:bool=False) -> None:
    Balance.UreaUnit(self)
    df = self.df.copy()

    self.flowCols = ["Component Flow (kmol/h)", 
                    "Mass Flow (mtpd)", 
                    "Mass Flow (kg/h)"]
    
    nh3Flow = self.q(df.loc[df["Component"] == "NH3"
                          ]["Mass Flow (mtpd)"].values[0] *3*.8, 'mtpd')
        
    scalar = (self.targetFlow / nh3Flow).__round__(5).magnitude 
    
    df = self.df.copy()
    if verbose: print(f"60 minute batch, 100% conversion", df, sep='\n')
    for col in self.flowCols: df[col] = df[col] * scalar
    
    self.renameDict = {      
      "Component Flow (kmol/h)": "Component Flow (kmol/20-min-batch)",
      "Mass Flow (mtpd)": "Mass Flow (mtpd/20-min-batch)",
      "Mass Flow (kg/h)": "Mass Flow (kg/20-min-batch)",
    }

    if verbose: print(f"20 minute batch, 80% conversion scaled by {scalar}", 
          df.rename(columns=self.renameDict), sep='\n')
    
    scalar2 = ((self.targetFlow / 3).magnitude / df.loc[df["Component"] == "NH3"
                    ]["Mass Flow (mtpd)"].values[0]).__round__(5) 

    for col in self.flowCols: df[col] = df[col] * scalar2
    if verbose:
      print(f"20 minute batch, 80% conversion, with recycle, scaled additionally by {scalar2}", 
          df.rename(columns=self.renameDict), sep='\n')
    
    s = (scalar*scalar2).__round__(3)
    if verbose:
      print(f"Total scaledown with 20 min batch, 80% conversion with recyle: {s}")
    
    self.batchFlows, self.totalFlows = df.rename(columns=self.renameDict), self.df

  def UreaUnit(self, verbose:bool=False) -> None:
    Balance.OverallMaterialBalance(self)
    df = self.ombal.copy()

    for x in [[1, "NH2COONH4"], [1, "NH2CONH2"], [-1, "CO2"],
              [-1, "H2O"]]:
      
      nmol = df.loc[df["Component"] == "NH3"]['Component Flow (kmol/h)'].values[0]
      mw = self.q(self.subs[x[1]].mass, 'gram/mol')
      stoich = x[0]
      
      molFlow = (self.q(nmol / 2 * stoich, 'kmol/h')).__round__(3)
      massFlow = (molFlow * mw).to("mtpd").__round__(3)
      if x[1] == "H2O": x[1] = "H2O (amm carb recy)"
      df = pd.concat([df,
                      pd.DataFrame({
                            "Component": [x[1]],
                            "Stoch Coeff": [stoich],
                            "Molecular Weight (grams/mol)": [mw.magnitude],
                            "Component Flow (kmol/h)": [molFlow.magnitude],
                            "Mass Flow (mtpd)": [massFlow.magnitude],
                            "Mass Flow (kg/h)": [massFlow.to('kg/h').magnitude]
                      })])
    
    self.df = df.reset_index(drop=True)
    if verbose: print(self.df)

  def FuelBurner(self, verbose:bool=False) -> None:
    Balance.UreaUnit(self)
    df = self.ombal.copy()
    print(df.rename(columns=self.renameDict).round(3))

    nCO2 = self.q(df.loc[
            df["Component"] == "CO2"]["Component Flow (kmol/h)"]
            .values[0] *3 # *3 because 20 min batch basis
          ,'kmol/h').to('kmol/d').__round__(3).__abs__()
        
    mCO2 = (nCO2 * self.q(self.subs["CO2"].mass, 'gram/mol')).to('kg/d')
    
    lhv = (self.q(42, 'MJ/kg') * mCO2).to("GJ/d").__round__(0)
    hhv = (self.q(55, 'MJ/kg') * mCO2).to('GJ/d').__round__(0).magnitude

    water = (nCO2 * 2 * self.q(self.subs["H2O"].mass, 'gram/mol')
              ).to('mtpd').__round__(3)
    
    ngPrice = self.q(9.53119, 'dollar / million_BTU')

    if verbose: 
      print(
        f"Total mols of CO2 required {nCO2}",
        f"Mass of CO2 required: {mCO2.to('mtpd').__round__(3)}",
        f"Cost of CH4: {(ngPrice * lhv).to('dollar/day').__round__(3)}",
        f"Water generated by CH4 combustion: {water}",
        f"Natural gas heating values 42-55 (MJ/kg)",
        f"Heat generated: {lhv.magnitude} - {hhv} GJ/day",
        sep='\n')

if __name__ == "__main__":
  obal, fracConv = False, True
  fuel, urea = False, False

  x = Balance()
  if obal: x.OverallMaterialBalance(True)
  if fracConv: x.FractionalConversion(True)
  if urea: x.UreaUnit(True)
  if fuel: x.FuelBurner(True)
