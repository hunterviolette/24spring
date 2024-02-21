import pandas as pd
import pint


class q4:
  def __init__(self) -> None:
    uReg = pint.UnitRegistry(autoconvert_offset_to_baseunit = True)
    uReg.define('dollar = [currency] = $')
    self.q=uReg.Quantity

    self.g = self.q(9.81, 'm/s**2')
  
  @staticmethod
  def printr(x):
    print(round(x,2))

  def one(self):
    '''
    The Grand Coulee dam reservoir (Roosevelt Lake) has a depth of 380 ft (119m) 
    at the base of the dam. The hydrostatic pressure at the base of the dam is
    '''
    rho = self.q(1000,'kg/m**3')
    h = self.q(119, 'm')
    
    q4.printr(
      (rho * self.g * h).to('MPa')
    )

  def two(self):
    '''
    A geothermal power plant uses hot water from a production well at 140oC, 
    and rejects heat to surroundings from a cooling tower at 17oC.  
    The maximum theoretical efficiency of this power plant is
    '''

    q4.printr(1 - (self.q(17,'degC') / self.q(140, 'degC')))

  def three(self):
    '''
    A geothermal power plant produces 50 litre/s of water at a 
    temperature of 85°C for district heating in a nearby town. 
    Using a base temp of 30°C, calculate the thermal power, in MWth. 
    Assume water has a specific heat of 4.2 kJ/kg/K.
    '''
    flow = (self.q(50, 'L/s') * self.q(1e3, 'kg/m**3')).to('kg/s')
    dT = self.q(85 - 30, 'degC')
    cp = self.q(4.2, 'kJ/kg/degK')

    q4.printr(
      (flow * cp * dT).to('MW')
    )

  def four(self):
    '''
    A ground source heat pump requires 12 kW of power to provide 180000 BTU/h 
    of heat to a commercial hot water heating system. The coefficient of 
    performance (COP) of this system is
    '''
  
    q4.printr(
      (self.q(1.8e5, 'BTU/h') / self.q(12,'kW')).to_base_units()
    )

  def five(self):
    '''
    A ground source heat pump with an Energy Efficiency Ratio (EER) of 11.5 
    supplies 80,000 BTU/hr to a residential heating system. How much power is required?
    '''

    q4.printr(
      (self.q(8e4, 'BTU/h') / 11.5).to('W')
    )

  def six(self):
    '''
    A hydro plant has a head of 12m and a flow rate of 80 m3/s. 
    The turbine output is 8 MW. Assuming no head losses, 
    estimate the efficiency of the turbine.
    '''
    flow = self.q(80, 'm**3/s') * self.q(1e3, 'kg/m**3')
    head = self.q(12, 'm')

    powerIn = (flow * head * self.g).to('MW')

    q4.printr(self.q(8, 'MW') / powerIn)

x = q4()
x.one()
x.two()
x.three()
x.four()
x.five()
x.six()