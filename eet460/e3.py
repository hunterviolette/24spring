import pint
import math
from chempy import Substance

class e3:
  def __init__(self) -> None:
    uReg = pint.UnitRegistry(autoconvert_offset_to_baseunit=True)
    uReg.define('dollars = [currency]')
    self.q = uReg.Quantity
    
    self.g = self.q(9.81, 'm/s**2')
    self.nl_conv = self.q(1 / 22.4, 'mol/liter')
    self.h2mass = self.q(Substance.from_formula('H2').mass, 'gram/mol')

  @staticmethod
  def printr(x, rounding: int = 2): print(round(x, rounding))

  def one(self):
    '''
    A fuel cell produces 58.9 amps at 0.51 volts, 
    with an H2 flow rate of 0.31 nlpm. 
    Calculate the thermal efficiency, based on HHV.
    '''
    power = (self.q(58.9, 'A') * self.q(.51, 'V')).to('W')
    flow = self.q(.31, 'L/min') / self.q(22.4, 'L/mol') * self.h2mass
    hhv = self.q(39.39, 'kWh/kg')

    e3.printr((power / (flow * hhv)).to_base_units(), 4)

  def two(self):
    '''
    Determine the minimum hydrogen flow rate (in nlpm) 
    supplied to a single cell to produce 68 Amps of current.
    '''
    molsH = self.q(68, 'ampere') / self.q(96485, 'C/mol') / 2
    
    e3.printr(
      (molsH * self.q(22.4, 'L/mol')).to('L/min')
    , 6)
  
  def three(self):
    '''
    The purchase price for a condo is $87,000.  
    Competing investments offer 5% interest for 8 years.  
    What should the selling price of the condo be to be 
    comparable to other investments?
    '''
    principal = self.q(87000, 'dollars')
    rate = 0.05
    years = 8

    e3.printr(principal * (1 + rate) ** years)

  def four(self):
    '''
    A high-efficiency motor costs $7200 and has a life of 12 years. 
    Assuming an interest rate of 10%, what annual energy cost savings 
    will be necessary to justify this expense?
    '''
    cost = self.q(7200, 'dollars')
    life_years = 12
    discount_rate = 0.10

    savings_needed = cost / ((1 - (1 + discount_rate) ** -life_years) / discount_rate)
    e3.printr(savings_needed)

  def five(self):
    '''
    A heat pump costs $6500 more than an electric resistance heater, 
    but is expected to save $1300/year over 10 years. 
    What is the net present worth of the heat pump, 
    assuming a discount rate of 9%?
    '''
    cost_difference = self.q(6500, 'dollars')
    annual_savings = self.q(1300, 'dollars')
    life_years = self.q(10, 'years').magnitude
    discount_rate = 0.09

    cashflows = [-cost_difference] + [annual_savings] * life_years
    npv = sum(cf / (1 + discount_rate) ** t for t, cf in enumerate(cashflows))
    e3.printr(npv)

  def six(self):
    '''
    You invest $20,000 and after 15 years your 
    investment is worth $67,000. What is the rate of 
    return on your investment?
    '''
    initial_investment = self.q(20_000, 'dollars')
    final_value = self.q(67_000, 'dollars')
    years = 15

    rate_of_return = ((final_value / initial_investment) ** (1 / years)) - 1
    e3.printr(rate_of_return.magnitude * 100)

  def seven(self):
    '''
    Given a wind speed of 8.65 m/s at a height of 10m in a 
    field with tall grass, find the wind speed at a height of 90m. 
    Use alpha = 0.15.
    '''
    v_10m = self.q(8.65, 'm/s')
    height_ratio = 90 / 10  # Calculate the ratio of heights
    alpha = .15

    e3.printr(v_10m * (height_ratio ** alpha))

  def eight(self):
    '''
    Find the average power in the wind (W/m^2) assuming Rayleigh statistics 
    in an area with 6.8m/s average wind speeds. Assume an air density of 1.225 kg/m3.
    '''
    v = self.q(6.8, 'm/s')
    rho = self.q(1.225, 'kg/m**3') 

    # Calculate average power in the wind
    average_power = 0.5 * rho * v ** 3

    e3.printr(average_power.to('W/m**2') * 1.91, 2)

  def nine(self):
    '''
    A wind turbine with a blade diameter of 95m 
    is designed to produce 2.6 MW with a wind speed of 12 m/s. 
    Estimate the power coefficient for this wind turbine. 
    Assume an air density of 1.225 kg/m3.
    '''
    power_generated = self.q(2.6, 'MW')
    wind_speed = self.q(12, 'm/s')
    blade_diameter = self.q(95, 'm')
    air_density = self.q(1.225, 'kg/m**3')

    blade_radius = blade_diameter / 2
    swept_area = math.pi * blade_radius ** 2
    Cp = power_generated / (0.5 * air_density * swept_area * wind_speed ** 3)

    e3.printr(Cp.to_base_units(), 4)

  def ten(self):
    '''
    A wind turbine with a blade diameter of 70 m generates its 
    rated power at a wind velocity of 9 m/s. 
    Estimate the maximum theoretical power of this turbine. 
    Assume an air density of 1.225 kg/m3
    '''
    blade_diameter = self.q(70, 'm')
    rated_wind_speed = self.q(9, 'm/s')
    air_density = self.q(1.225, 'kg/m**3')

    blade_radius = blade_diameter / 2
    swept_area = math.pi * blade_radius ** 2
    max_theoretical_power = 0.5 * air_density * swept_area * rated_wind_speed ** 3

    e3.printr(max_theoretical_power.to('MW'))

  def eleven(self):
    '''
    The Sihwa Lake tidal power station reservoir has a surface area of 30 km**2. 
    Using an average tidal range of 5.6 m, estimate the energy of this lake as a 
    tidal resource, in GWh/yr.
    '''
    rho = self.q(1025, 'kg/m**3')
    area = self.q(30, 'km**2')
    reservoir  = self.q(5.6, 'm')
    cycles = self.q(706.5, '1/year')

    te = ( # tidal energy
            rho * self.g
            * area * reservoir**2 / 2
          ).to('J')

    e3.printr((te * cycles).to('GW*h/year'), 2)

x = e3()
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
x.eleven()