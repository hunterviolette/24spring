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
  6HCL + 6NH3 -> 6NH4Cl

Main rxn:
  Mg3N2 + 6NH4Cl = 3MgCl2 + 8NH3

Overall:
  3MgCl2 + N2 + 3H2O -> 2NH3 + 1.5O2

'''

class Balance: 
  def __init__(self) -> None:
    uReg = pint.UnitRegistry(autoconvert_offset_to_baseunit = True)
    uReg.define('mtpd = metric_ton / day')
    self.q = uReg.Quantity

    self.subs = {
      'MgCl2': Substance.from_formula('MgCl2'),
      'Mg': Substance.from_formula('Mg'),
      'Cl2': Substance.from_formula('Cl2'),
      'N2': Substance.from_formula('N2'),
      'H2O': Substance.from_formula('H2O'),
      'HCl': Substance.from_formula('HCl'),
      'NH3': Substance.from_formula('NH3'),
      'NH4Cl': Substance.from_formula('NH4Cl'),
      'Mg3N2': Substance.from_formula('Mg3N2'),
      'O2': Substance.from_formula('O2')
    }

  def OverallMaterialBalance(self):

    targetFlow = self.q(10, 'mtpd')
    targetMW = self.q(self.subs["NH3"].mass, 'gram/mol')

    molph = (targetFlow / targetMW).to('kmol/h')
    
    # rxn stoich
    ## 3 MgCl2 + N2 + 3 H2O -> 2 NH3 + 1.5 O2 + 3 MgCl2
    
    df = pd.DataFrame()
    for x in [[-3, "MgCl2"], [-1, "N2"], [-3, "H2O"],
              [2, "NH3"], [1.5, "O2"], [3, "MgCl2"]]:
      
      mw = self.q(self.subs[x[1]].mass, 'gram/mol')
      stoich = x[0]

      molFlow = (molph / 2 * stoich).__round__(3)
      massFlow = (molFlow * mw).to("kg/h").__round__(3)

      df = pd.concat([df,
                      pd.DataFrame({
                            "Component": [x[1]],
                            "Stoch Coeff": [stoich],
                            "Molecular Weight (grams/mol)": [mw.magnitude],
                            "Component Flow (kmol/h)": [molFlow.magnitude],
                            "Mass Flow (kg/h)": [massFlow.magnitude]
                      })])
      
    self.ombal = df.reset_index(drop=True) # Overall material balance
    print('=== ===',
          f"Target: {targetFlow}",
          f"NH3 mass per day: {(molph*targetMW).to('kg/day').__round__(4)}",
          f"NH3 mass per hour: {(molph*targetMW).to('kg/h').__round__(4)}",
          f"NH3 mols/h: {molph.__round__(4)}",
          '=== Overall Material Balance ===',
          self.ombal,
          '',
          sep='\n\n')
    
  def ReactorVolume(self):
    Balance.OverallMaterialBalance(self)
    col = "Mass Flow (kg/h)"
    
    catalyst = self.q(self.ombal.at[5, col], 'kg/h')
    nh3 = self.q(self.ombal.at[3, col] * 4, 'kg/h')

    molFlow = self.q(self.ombal.at[3, 'Component Flow (kmol/h)'], 'kmol/h')

    nh4cl = (molFlow * 6 / 8 * 
              self.q(self.subs["NH4Cl"].mass, 'gram/mol')
            ).to('kg/h')
    
    mg3n2 = (molFlow * 3 / 8 * 
              self.q(self.subs["Mg3N2"].mass, 'gram/mol')
            ).to('kg/h')

    nh3syn = self.q(10**-6, 'mol/cm**2/s')

    print('=== Reactor Inputs ===', 
          f'NH4Cl (g): {nh4cl.__round__(3)}',
          f"Mg3N2 (s): {mg3n2.__round__(3)}",
          '=== Reactor Outputs ===',
          f'MgCl2 (l): {catalyst}',
          f"NH3 (g): {nh3}",
          '=== NH3 Production ===',
          f"NH3 (total produced): {nh3.to('mtpd').__round__(3)}",
          f"NH3 (stored): {(nh3 * 2 / 8).to('mtpd').__round__(3)}",
          '=== Reactor Area ===',
          f"NH3 Synthesis: {nh3syn}",
          f"Area (net): {(molFlow / nh3syn).to('m**2').__round__(3)}",
          f"Area (total): {(molFlow / nh3syn * 4).to('m**2').__round__(3)}",
          '',
          sep='\n\n')

if __name__ == "__main__":

  x = Balance()
  x.ReactorVolume()