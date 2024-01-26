
import pandas as pd
import pint
import cmath
import math

class q1:
  def __init__(self) -> None:
    self.q = pint.UnitRegistry(autoconvert_offset_to_baseunit = True).Quantity

  def one(self):

    sol = (self.q(50, 'hertz') * self.q(2, 'pi')).to('rad/s') 
    print(sol.__round__(2))

  def two(self):
    sol = 7200 / 240
    print(sol)

  def three(self):
    z1 = cmath.rect(2, math.radians(45)) 
    z2 = cmath.rect(8, math.radians(-15))  
    print(z1*z2)

  def four(self):
    i = cmath.rect(20, math.radians(-35))
    i2 = cmath.rect(40, math.radians(55))

    magnitude, angle = cmath.polar(i+i2)

    print(magnitude, math.degrees(angle))

  def five(self):

    x = self.q(10,'kW') / self.q(12,'kV*A')
    print(x.to_base_units())

  def six(self):
    power = self.q(500, 'W') 
    voltage = self.q(120,'volts')  

    current = power / voltage
    ohms = (voltage / current).to('ohm')
    print(round(ohms,2))

  def seven(self):
    power = self.q(5, 'kW') 
    voltage = self.q(120,'volts')  
    amps = self.q(10,'A')

    supplyPower = (voltage * amps).to('kW')
    turnsRatio = power / supplyPower
    print(turnsRatio)

if __name__ == "__main__":
  x = q1()
  x.one()
  x.two()
  x.three()
  x.four()
  x.five()
  x.six()
  x.seven()