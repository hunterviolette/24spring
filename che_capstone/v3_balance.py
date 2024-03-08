import pandas as pd
import pint

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

class Balance: 
  def __init__(self) -> None:
    uReg = pint.UnitRegistry(autoconvert_offset_to_baseunit = True)
    uReg.define('mtpd = metric_ton / day')
    self.q = uReg.Quantity

    self.targetFlow = self.q(1, 'mtpd')

    self.subs = {
      'MgCl2': Substance.from_formula('MgCl2'),
      'Mg': Substance.from_formula('Mg'),
      'Cl2': Substance.from_formula('Cl2'),
      'N2': Substance.from_formula('N2'),
      'H2O': Substance.from_formula('H2O'),
      'HCl': Substance.from_formula('HCl'),
      'NH3': Substance.from_formula('NH3'),
      'Mg3N2': Substance.from_formula('Mg3N2'),
      'O2': Substance.from_formula('O2')
    }

  def OverallMaterialBalance(self, verbose:bool=False):

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

  def FractionalConversion(self):
    Balance.OverallMaterialBalance(self)
    df = self.ombal.copy()

    self.flowCols = ["Component Flow (kmol/h)", 
                    "Mass Flow (mtpd)", 
                    "Mass Flow (kg/h)"]
    
    nh3Flow = self.q(df.loc[df["Component"] == "NH3"
                          ]["Mass Flow (mtpd)"].values[0] *3*.8, 'mtpd')
        
    scalar = (self.targetFlow / nh3Flow).__round__(5).magnitude 
    
    df = self.ombal.copy()
    print(f"60 minute batch, 100% conversion", df, sep='\n')
    for col in self.flowCols: df[col] = df[col] * scalar
    
    renameDict = {      
      "Component Flow (kmol/h)": "Component Flow (kmol/20-min-batch)",
      "Mass Flow (mtpd)": "Mass Flow (mtpd/20-min-batch)",
      "Mass Flow (kg/h)": "Mass Flow (kg/20-min-batch)",
    }

    print(f"20 minute batch, 80% conversion scaled by {scalar}", 
          df.rename(columns=renameDict), sep='\n')
    
    scalar2 = (.3334 / df.loc[df["Component"] == "NH3"
                    ]["Mass Flow (mtpd)"].values[0]).__round__(5) 

    for col in self.flowCols: df[col] = df[col] * scalar2
    print(f"20 minute batch, 80% conversion, with recycle, scaled additionally by {scalar2}", 
          df.rename(columns=renameDict), sep='\n')
    
    s = (scalar*scalar2).__round__(3)
    print(f"Total scaledown with 20 min batch, 80% conversion with recyle: {s}")

if __name__ == "__main__":

  x = Balance()
  x.FractionalConversion()