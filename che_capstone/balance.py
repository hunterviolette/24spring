import pandas as pd
import pint

uReg = pint.UnitRegistry(autoconvert_offset_to_baseunit = True)
#uReg.default_format = "~P"

uReg.define('mtpd = metric_ton / day')

q = uReg.Quantity

'''
Reactions:

N2 + 3H2 <-> 2NH3
H2 + 1/2 O2 -> H2O
'''

class Balanace:
  def __init__(self, 
              prod_flow_rate: float = 10.0, # in metric tonne per day
              verbose: bool = False
            ) -> None:
    
    self.verbose = verbose
    self.prod_flow_rate = q(prod_flow_rate, 'mtpd')
    
    self.molec_weights = {
      "H2": q(1.008 * 2, 'grams/mol'),
      "O2": q(15.999 * 2, 'grams/mol'),
      "N2": q(14.007 * 2, 'grams/mol'),
      "H2O": q(1.008 * 2 + 15.999, 'grams/mol'),
      "NH3": q(1.008 * 3 + 14.007, 'grams/mol'),
      "AIR": q(28.9647, 'grams/mol') # Assumes dry air
    }

  def MoleFlow(self):
    self.mol_flow = {
      "NH3": (self.prod_flow_rate / self.molec_weights["NH3"]).to('kmol/day')
    }

    self.mol_flow["N2"] = self.mol_flow["NH3"] / 2
    self.mol_flow["H2"] = self.mol_flow["NH3"] / 2 * 3
    self.mol_flow["AIR"] = self.mol_flow["N2"] / .7808
    self.mol_flow["H2O"] = self.mol_flow["H2"]

    if self.verbose:
      print(f'=== Mole Flows, basis: {self.prod_flow_rate} ===')
      for x in self.mol_flow.keys():
        print(f"{x} {self.mol_flow[x].__round__(2)}")

  def MassFlow(self):
    Balanace.MoleFlow(self)
    
    self.mass_flow = {}
    for key in self.mol_flow.keys():
      self.mass_flow[key] = (self.mol_flow[key] * self.molec_weights[key]).to('kg/day')

    if self.verbose:
      print(f'=== Mass Flows ===')
      for x in self.mass_flow.keys():
        print(f"{x} {self.mass_flow[x].__round__(2)}")
        
  def NormalizedVdot(self):
    Balanace.MassFlow(self)
    
    # PV = nRT
    p = q(1, 'atm')
    t = q(0, 'degC').to('degK')
    r = q(8.3144621, 'J/mol/degK')
  
    self.normVdot = {}
    for key in self.mol_flow.keys():
      self.normVdot[key] = (self.mol_flow[key] * r * t / p
                              ).to('m**3/h').__round__(2)

    print(
      f"Hydrogen: {self.normVdot['H2']}",
      f"Nitrogen: {self.normVdot['N2']}",
      f"Ammonia: {self.normVdot['NH3']}",
      f"Ammonia Mass Flow: {self.mass_flow['NH3'].to('mtpd')}",
      sep='\n'
    )
    

if __name__ == "__main__":
  Balanace(verbose=False).NormalizedVdot()