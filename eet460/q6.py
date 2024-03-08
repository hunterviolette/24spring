import pandas as pd
import pint

from chempy import Substance


class q4:
  def __init__(self) -> None:
    uReg = pint.UnitRegistry(autoconvert_offset_to_baseunit = True)
    uReg.define('dollar = [currency] = $')
    self.q=uReg.Quantity

    self.g = self.q(9.81, 'm/s**2')
    self.h2mass = self.q(Substance.from_formula('H2').mass, 'gram/mol')
  @staticmethod
  def printr(x, rounding: int = 2):
    print(round(x,rounding))

  def one(self):
    '''
    Determine the minimum hydrogen flow rate (in nlpm) 
    supplied to a single cell to produce 25 Amps of current.
    '''
    molsH = self.q(15, 'ampere') / self.q(96485, 'C/mol') / 2
    
    q4.printr(
      (molsH * self.q(22.4, 'L/mol')).to('L/min')
    , 6)

  def two(self):
    '''
    Calculate the mass of 300 liters of Hydrogen 
    at a pressure of 200 Bar and a temperature of 25°C.
    '''
    p = self.q(200, 'bar')
    v = self.q(300, 'L')
    t = self.q(25, 'degC')
    r = self.q(1, 'molar_gas_constant')

    n = p*v / (r * t)

    q4.printr(
      (n.to('mol') * 
      (self.h2mass)
      ).to('kg')
    , 6)

  def three(self):
    '''
    A fuel cell produces 16 Amps at 0.50 volts, 
    with an H2 flow rate of 0.11 nlpm. 
    Calculate the thermal efficiency, based on HHV.
    '''
    power = (self.q(16, 'A') * self.q(.5, 'V')).to('W')
    flow = self.q(.11, 'L/min') / self.q(22.4, 'L/mol') * self.h2mass
    hhv = self.q(39.39, 'kWh/kg')

    q4.printr((power / (flow * hhv)).to_base_units())

  def four(self):
    '''
    Calculate the Gibbs Free Energy for the following hydrogen 
    reaction at a temperature of 800°C.

    H2 + ½O2 → H2O(g)

    Note: at this temperature, the enthalpy of reaction is -249.5 kJ/mol, 
    and the change in entropy is -57.0 J/mol·K.
    '''
    dH = self.q(-249.5, 'kJ/mol')
    t = self.q(800, 'degC')
    dS = self.q(-57, 'J/mol/K')
    q4.printr((dH - t*dS).to('kJ/mol'))

  def five(self):
    '''
    Calculate the lower heating value (LHV) for the following ethanol reaction. 
    Assume standard conditions (p = 1atm, T = 25°C). 
    The molecular weight of ethanol is 46.07 g/mole.
    '''
    eth = self.q(-277.7, 'kJ/mol')
    o2 = self.q(0, 'kJ/mol')
    co2 = self.q(-393.5, 'kJ/mol')
    h2o = self.q(-241.8, 'kJ/mol')

    dH = 3*h2o + 2*co2 - eth - o2
    
    q4.printr((dH / self.q(Substance.from_formula('C2H5OH').mass, 'gram/mol')).to('MJ/kg'))

x = q4()

x.one()
x.two()
x.three()
x.four()
x.five()