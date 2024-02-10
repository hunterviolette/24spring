import pandas as pd
import pint
import cmath
import math

class e1:
  def __init__(self) -> None:
    uReg = pint.UnitRegistry(autoconvert_offset_to_baseunit = True)
    uReg.define('dollar = [currency] = $')
    self.q=uReg.Quantity
  
  def one(self):
    '''
    A system has 10kVA and 8kW. What is the PF?
    '''
    print(
      (self.q(8,'kW') / self.q(10, 'kV*ampere')).to_base_units()
    )

  def two(self):
    '''
    A power plant operates at full output for 
    4000hours in a year. What is the capacity factor?
    '''
    print(
      (self.q(4000,'h') / self.q(1,'year')).to_base_units()
    )

  def three(self):
    '''
    A single phase 7200V distribution line at with a current 
    draw of 10A feeds a large commercial facility. The commercial 
    facility is being fed via a 7200V/240V transformer. What is 
    the magnitude of current going into the panelboard in the facility?
    '''
    voltIn = self.q(7200, 'volt')
    currentIn = self.q(10, 'ampere')
    voltOut = self.q(240, 'volt')
    
    print(
      (voltIn *  currentIn / voltOut).to_base_units()
    )

  def four(self):
    '''
    A gas turbine generator has a heat rate of 10,900 BTU/kWh. 
    The efficiency of this power plant is most nearly

    '''
    print(
      1 / self.q(10900, 'BTU/(kWh)').to_base_units() * 100
    )
    
  def five(self):
    '''
    A steam power plant produces 900MW of electricity 
    at an efficiency of 37%. How much heat is rejected to the surroundings?
    '''
    print(
      (self.q(900,'MW') / .37) - self.q(900,'MW')
    )

  def six(self):
    '''
    A steam power plant has a maximum steam temperature 
    of 1050°F, and the power plant rejects heat to the 
    surroundings at an average temperature of 59°F. 
    What is the maximum possible efficiency of this power plant?
    '''
    print(
      1 - (self.q(59,'degF') / self.q(1050, 'degF'))
    )

  def seven(self, 
            capacityFactor: float = .4, 
            fixChargeRate: float = .1
          ):
    '''
    Given a combined cycle power plant with a capacity factor 
    of 0.4, and a fixed charge rate of 10%, what is the 
    unitized cost of the plant. Use the following data:

    Capital Cost = $600/kW
    Heat Rate = 7700 BTU/kWh
    Fuel Cost = $4.50/MMBTU
    Variable O&M Costs = $0.0037/kWh
    '''
    capCost = self.q(600,'dollar/kW')
    heatRate = self.q(7700,'BTU/kWh') 
    fuelCost = self.q(4.50,'dollar/MBTU') 
    varOM = self.q(0.0037,'dollar/kWh')

    fixedCost = (fixChargeRate * capCost / 
                (capacityFactor * self.q(1,'year'))).to('dollar/kWh')
    
    varCost = (fuelCost * heatRate).to('dollar/kWh').__round__(3)

    print(
      '====',
      f"Fix charge rate: {fixChargeRate} and capacity factor: {capacityFactor}",
      f"fixed cost: {fixedCost}",
      f"variable cost: {varCost}",
      (fixedCost + varCost + varOM).to('dollar/(kW*h)'),
      (fixedCost + varCost + varOM).to('dollar/(kW*year)'),
      sep='\n'
    )

x = e1()

x.one()
x.two()
x.three()
x.four()
x.five()
x.six()

x.seven(capacityFactor=.5, fixChargeRate=.14)
x.seven(capacityFactor=.4, fixChargeRate=.1)