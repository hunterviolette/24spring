import pint

from scipy import optimize
from chempy import Substance
import math
import numpy as np

class e2:
  def __init__(self) -> None:
    uReg = pint.UnitRegistry(autoconvert_offset_to_baseunit = True)
    uReg.define('dollar = [currency] = $')
    self.q=uReg.Quantity

    self.g = self.q(9.81, 'm/s**2')
    self.h2mass = self.q(Substance.from_formula('H2').mass, 'gram/mol')
  
  @staticmethod
  def pprnt(x, rounding: int = 2):
    print(round(x,rounding))

  def one(self):
    '''
    A 900 MW nuclear power plant supplies 7210 GWH/yr. 
    Calculate the capacity factor
    '''

    theoOutput = (self.q(900, 'MW') * self.q(1,'yr').to('h') / self.q(1,'yr')
                ).to('GWh/yr')
    e2.pprnt(
      (self.q(7210, "GWh/yr") / theoOutput).to_base_units()
    , rounding=5)

  def two(self):
    '''
    A 60 MW waste-to-energy plant processes 1910 metric tons of municipal solid waste per day, 
    with a heating value (HHV) of 11,070 kJ/kg. 
    Calculate the efficiency of this plant.
    '''

    power_output = self.q(60, 'MW')  
    waste = self.q(1910, 'tonne')  
    hhv = self.q(11070, 'kJ/kg')  

    eInput = (waste * hhv).to('kJ')
    eOutput = (power_output * self.q(24, 'h')).to('kJ')

    eff = (eOutput / eInput).to_base_units()
    e2.pprnt(eff, 5)

  def three(self):
    '''
    A nuclear power plant produces 875,000 kW of electricity while rejecting 1480 MW of waste heat to the surroundings.
    Calculate the efficiency of this plant.
    '''

    electricity_output = self.q(875000, 'kW') 
    heat_rejected = self.q(1480, 'MW') 

    energy_input = (electricity_output + heat_rejected).to('kW')
    efficiency = (electricity_output / energy_input).to_base_units()

    e2.pprnt(efficiency, rounding=5)

  def four(self):
    '''
    A ground source heat pump requires 9.4 kW to provide 80,000 BTU/hr to a residential heating system.
    The Energy Efficiency Ratio (EER) of this system is.
    '''

    power_input = self.q(9.4, 'kW')
    cooling_capacity = self.q(80000, 'BTU/h')

    eer = (cooling_capacity / power_input).to_base_units()

    e2.pprnt(eer, rounding=5)

  def five(self):
    '''
    A geothermal power plant produces 150 litre/s of water at a 
    temperature of 83°C for district heating in a nearby town. 
    Using a base temp of 30°C, calculate the thermal power, in MWth. 
    Assume water has a specific heat of 4.2 kJ/kg/K.
    '''

    density_water = self.q(1000, 'kg/m^3')
    volume_flow_rate = self.q(150, 'L/s')
    outlet_temp = self.q(83, 'degC')
    base_temp = self.q(30, 'degC')
    specific_heat_water = self.q(4.2, 'kJ/kg/K')

    mass_flow_rate = (volume_flow_rate * density_water).to('kg/s')
    thermal_power = (mass_flow_rate * specific_heat_water * (outlet_temp - base_temp)).to('MW')

    e2.pprnt(thermal_power)

  def six(self):
    '''
    A ground source heat pump with a coefficient of performance (COP) of 4.5 supplies 120,000 BTU/hr 
    of heat to a residential hot water heating system. How much power is required?
    '''

    cop = 4.5
    heating_output = self.q(120000, 'BTU/hr')  # Heating output in BTU/hr
    power_input = (heating_output / cop).to('kW')

    e2.pprnt(power_input)

  def seven(self):
    power_output = self.q(4500, 'kW')  
    fuel_consumption = self.q(215, 'g/kWh').to('kg/kWh')
    lhv_diesel = self.q(42.75, 'MJ/kg')  

    energy_input = (fuel_consumption * power_output * lhv_diesel)
    efficiency = ((power_output / energy_input)).to_base_units()

    e2.pprnt(efficiency, rounding=5)

  def eight(self):
    density_water = self.q(1000, 'kg/m**3')
    flow_rate = self.q(68, 'm**3/s')
    head = self.q(78, 'm')
    power_output = self.q(45, 'MW')

    power_input = (density_water * self.g * flow_rate * head)
    efficiency = (power_output / power_input).to_base_units()

    e2.pprnt(efficiency, rounding=5)

  def nine(self):
    electrical_power = self.q(600, 'kW')  
    gas_input = self.q(5.9, 'MBTU/h')
    thermal_output = self.q(2.4, 'MBTU/h')

    total_output = (electrical_power + thermal_output)
    efficiency = (total_output / gas_input).to_base_units()

    e2.pprnt(efficiency, rounding=5)

  def ten(self):
    # Process fluid is water
    reservoir_elevation = self.q(326, 'm')
    penstock_length = self.q(6.4, 'km')
    power_plant_elevation = self.q(18, 'm')
    inlet_pressure = self.q(2.4, 'MPa')
    density = self.q(1000, 'kg/m**3')
    diameter = self.q(5, 'm') 

    # Yaws critical properties Table 82. Viscosity of Liquid 5C
    dynamic_viscosity = self.q(1.468, 'cP') 

    head_difference = reservoir_elevation - power_plant_elevation

    # Calculate fluid velocity using the Bernoulli equation
    fluid_velocity = (self.q(math.sqrt((
                        (2 * inlet_pressure.to('Pa') / density)
                        + 2 * self.g * head_difference
                      ).to("m**2/s**2").magnitude
                    ), 'm/s'))

    reynolds_number = (density * fluid_velocity 
                       * diameter / dynamic_viscosity
                      ).to_base_units().__round__(3)

    # Calculate friction factor using Colebrook equation
    def colebrook(f, Re, D):
      return (-2 * np.log10((2.51 / (Re * np.sqrt(f))) 
              + (f / (3.71 * D))) - 1.0 / np.sqrt(f))
    
    initial_guess = 0.001  # Initial guess for friction factor
    friction_factor =optimize.newton(colebrook, initial_guess, maxiter=1000,
                    args=(reynolds_number.magnitude, diameter.magnitude))
    

    # Calculate the head losses using Darcy-Weisbach equation    
    penstock_head_loss = (
        (friction_factor * penstock_length* fluid_velocity**2 
          / (2 * self.g * diameter))
      ).to('m')
    
    vars = {
        "high elevation": reservoir_elevation,
        "low elevation": power_plant_elevation,
        "penstock lenth": penstock_length,
        "turbine inlet pressure": inlet_pressure,
        "process fluid density": density,
        "pipe diameter": diameter,
      }
    
    print("===== Problem 10 =====")
    print("=== Input variables ===")
    for var in vars.keys(): 
      print(f"{var}: {vars[var].__round__(3)}")

    vars = {
        "change in elevation": head_difference,
        "friction factor": friction_factor,
        "fluid velocity": fluid_velocity,
        "dynamic viscosity": dynamic_viscosity,
        "Reynold's number": reynolds_number,
        "head loss": penstock_head_loss,
      }
    
    print("=== Export variables ===")
    for var in vars.keys(): 
      print(f"{var}: {vars[var].__round__(3)}")
  
x = e2()

x.one()
x.two()
x.three()
x.four()
x.five()
x.six()
x.seven()
x.eight()
x.nine()
x.ten()