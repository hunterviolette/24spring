import pint
import math

class q9:
  def __init__(self) -> None:
      uReg = pint.UnitRegistry(autoconvert_offset_to_baseunit=True)
      uReg.define('dollars = [currency]')
      self.q = uReg.Quantity
      
      self.g = self.q(9.81, 'm/s**2')
      self.nl_conv = self.q(1 / 22.4, 'mol/liter')

  @staticmethod
  def printr(x, rounding: int = 2):
    print(round(x, rounding))

  def one(self):
    '''
    A sea state has a significant wave height of 
    3.75 meters and a wave period of 8.5 seconds. 
    The wave power density is most nearly:
    '''
    rho = self.q(1025, 'kg/m**3')
    h = self.q(3.75, 'm')
    period = self.q(8.5, 's')

    # wave power density 
    wpd = (
            rho * self.g**2 
            * h**2 * period 
            / (64 * math.pi)
          )
    
    q9.printr(wpd.to('kW/m'), 2)

  def two(self):
    '''
    The Jiangxia tidal power station reservoir 
    has a surface area of 137 km**2. Using an 
    average tidal range of 8.39 m, estimate the 
    energy of this lagoon as a tidal resource, in GWh/yr.
    '''
    rho = self.q(1025, 'kg/m**3')
    area = self.q(137, 'km**2')
    reservoir  = self.q(8.3, 'm')
    cycles = self.q(706.5, '1/year')

    # tidal energy
    te = (
            rho * self.g
            * area * reservoir**2 / 2
          ).to('J')

    q9.printr((te * cycles).to('GW*h/year'), 2)

  def three(self):
    '''
    A tidal turbine with a blade diameter of 12 meters generates 75 kW 
    of electricity in a tidal stream with a velocity of 1.5 m/s. 
    Assume a water density of 1000kg/m3. 
    The efficiency of this tidal turbine is most nearly:
    '''
    diameter = self.q(12, 'm')
    power = self.q(75, 'kW')
    velocity = self.q(1.5, 'm/s')
    rho = self.q(1025, 'kg/m**3')

    # power coefficient
    Cp = (2 * power) / (diameter**2 * rho * velocity**3)

    q9.printr(Cp.to_base_units(), 4)

  def four(self):
    '''
    A battery system is rated at 10 MW capacity, with a 
    total usable storage capability of 80 MWh. How long 
    can this system operate at its design capacity?
    '''
    capacity = self.q(10, 'MW')
    total_storage = self.q(80, 'MWh')

    time = total_storage / capacity

    q9.printr(time.to('h'), 3)

  def five(self):
    '''
    A commercial electrolyzer produces 60 Nm**3/hr 
    of hydrogen with an electrical power input of 294 kW. 
    Determine the efficiency of this electrolyzer (HHV basis).
    '''
    
    powerIn = self.q(294, 'kW')
    vflow = self.q(60, 'm**3/h')
    
    mol_flow = self.q(vflow.to('m**3/h').magnitude *1e3, 'L/hr') * self.nl_conv 

    powerOut = mol_flow * self.q(285.8, 'kJ/mol')

    q9.printr((powerOut / powerIn).to_base_units(), 2)

x = q9()
x.one()
x.two()
x.three()
x.four()
x.five()
