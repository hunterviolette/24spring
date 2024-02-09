import pandas as pd
import pint
import cmath
import math

class q3:
  def __init__(self) -> None:
    uReg = pint.UnitRegistry(autoconvert_offset_to_baseunit = True)
    uReg.define('dollar = [currency] = $')
    self.q=uReg.Quantity
  
  def one(self):
    '''
    A 60 MW solar PV power plant produces 110 GWh/yr. 
    Determine the equivalent hours of operation per year at rated power.
    '''
    print((self.q(110, 'GWh/yr') / self.q(60, 'MW')).to('h/yr'))

  def two(self):
    '''
    A power plant owner pays $7million per year for a power plant. 
    With a fixed charge rate of 12%, what is the cost of construction for this plant?
    '''
    print(7e6/.12)

  def three(self):
    '''
    A gas turbine engine has an airflow rate of 981 lb/s 
    and an exhaust temperature of 1120 degF. Using a minimum stack temperature of 220 degF, 
    how much thermal power is available in the exhaust? Use Cp = 0.24 BTU/lb-degR.
    '''
    massFlow = self.q(981,'lb/s')
    tOut = self.q(1120,'degF')
    tIn = self.q(220, 'degF')
    cp = self.q(.24, 'BTU/lb/degR')

    print((massFlow*cp*(tOut-tIn)).to('MBTU/h'))

  def four(self):
    '''
    A combined cycle power plant is comprised of two 85 MW gas turbines 
    and one 90 MW steam turbine. The turbines run on natural gas, 
    and the exhaust supplies energy to the steam cycle through an HRSG. 
    While running at full capacity, the plant consumes 1.67 million scf
    of natural gas per hour. Using a heating value (HHV) for natural gas
    of 1021 BTU/scf, determine the heat rate of this plant.
    '''
    fixChargeRate = .14
    capacityFactor = .5
    variableCost = self.q(.0037, 'dollar/kWh')

    fc = (fixChargeRate * self.q(600, 'dollar/kW') / \
          (capacityFactor * self.q(1,'year'))).to('dollar/kWh')
    
    vc = (self.q(7700, 'BTU/(kW*h)') * self.q(4.5, 'dollar/MBTU')).to('dollar/kWh')
    
    print(vc+fc+variableCost)

  def five(self):
    '''
    Given the following information on two power plants

    Coal plant: Fixed Cost = $180/kW Variable Cost = $0.0181/kWh
    Combustion Turbine: Fixed Cost = $48/kW Variable Cost = $0.0575/kWh

    Which of the following statements is true?
    '''

    coalfc = self.q(180, 'dollar/kW')
    coalvc = self.q(.0181, 'dollar/kWh')

    turfc = self.q(48, 'dollar/kW')
    turvc = self.q(.0575, 'dollar/kWh')

    for hours in [3350, 1746, 3016, 835]:
      print(f'--{hours}--')
      for i,x in enumerate([[coalfc, coalvc], [turfc, turvc]]):
        if i == 0: process = 'coal'
        else: process = 'turbine'
        cost = ((x[1] * self.q(hours, 'h')).to('dollar/kW') + x[0]).__round__(0)

        print(f"cost for {process}: {cost}")

  def six(self):
    turbineOutput = self.q(85*2 + 90, 'MW')
    heatRate = self.q(1.67e6, 'ft**3/h') * self.q(1021, 'BTU/ft**3')

    print((heatRate/turbineOutput).to('BTU/kWh'))

q3().five()