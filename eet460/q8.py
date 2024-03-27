import pint
import math

class q8:
  def __init__(self) -> None:
    uReg = pint.UnitRegistry(autoconvert_offset_to_baseunit=True)
    uReg.define('dollars = [currency]')
    self.q = uReg.Quantity

  @staticmethod
  def printr(x, rounding: int = 2):
    print(round(x, rounding))

  def one(self):
    '''
    Wind velocity measurements over a period of one year yield a 
    frequency distribution with an average wind velocity of 4.7 m/s. 
    We wish to model this data with a Rayleigh distribution function. 
    The Rayleigh distribution function parameters will be:
    '''
    v = self.q(4.7, 'm/s')
    k = 2  # For Rayleigh distribution, k is always 2

    q8.printr(v / (2 ** 0.5), 4)
  
  def two(self):
    '''
    An anemometer mounted at a height of 10m on a level field shows an average windspeed of 6m/s. 
    Using a value of alpha = 1/7, estimate the average windspeed at a height of 80m.
    '''
    v_10m = self.q(6, 'm/s')
    height_ratio = 80 / 10  # Calculate the ratio of heights

    # Given alpha value
    alpha = 1 / 7

    # Estimate average windspeed at 80m
    v_80m = v_10m * (height_ratio ** alpha)

    q8.printr(v_80m, 2)

  def three(self):
    '''
    Find the average power in the wind (W/m^2) assuming Rayleigh statistics 
    in an area with 7.5m/s average wind speeds. Assume an air density of 1.225 kg/m3.
    '''
    v = self.q(7.5, 'm/s')
    rho = self.q(1.225, 'kg/m**3') 

    # Calculate average power in the wind
    average_power = 0.5 * rho * v ** 3

    q8.printr(average_power.to('W/m**2'), 2)

  def four(self):
    '''
    A wind turbine with a blade diameter of 40 m generates its rated power 
    at a wind velocity of 12 m/s. Estimate the maximum 
    theoretical power of this turbine. Assume an air density of 1.225 kg/m3.
    '''
    wind_speed = self.q(12, 'm/s')
    blade_diameter = self.q(40, 'm')
    air_density = self.q(1.225, 'kg/m^3')

    # Calculate radius from diameter
    blade_radius = blade_diameter / 2

    # Calculate angular velocity (omega) using rated power condition
    rated_power_wind_speed = wind_speed
    omega = (rated_power_wind_speed / blade_radius).to_base_units()

    '''
    # Calculate tip-speed ratio (TSR)
    TSR = (omega * blade_radius) / wind_speed

    # Calculate power coefficient (Cp)
    Cp = (16 / 27) * (math.sin(TSR) / (1 - math.cos(TSR))) ** 2
    '''

    # Calculate maximum theoretical power
    max_theoretical_power = 0.5 * air_density * math.pi * blade_radius ** 2 * wind_speed ** 3

    q8.printr(max_theoretical_power.to('MW'))

  def five(self):
    '''
    A wind turbine with a blade diameter of 52m is designed to 
    produce 825kW with a wind speed of 12.5m/s. 
    Estimate the power coefficient for this wind turbine.
    '''
    power_generated = self.q(825, 'kW')
    wind_speed = self.q(12.5, 'm/s')
    blade_diameter = self.q(52, 'm')
    air_density = self.q(1.225, 'kg/m^3')

    # Calculate radius from diameter
    blade_radius = blade_diameter / 2

    # Calculate swept area
    swept_area = math.pi * blade_radius ** 2

    # Calculate power coefficient
    Cp = power_generated / (0.5 * air_density * swept_area * wind_speed ** 3)

    q8.printr(Cp.to_base_units())

  def six(self):
    '''
    A wind turbine with a blade diameter of 58 m has a rotor speed of 
    30.8 rpm with a windspeed of 11.5 m/s. 
    Calculate the Tip Speed Ratio (TSR) at this condition.
    '''
    blade_diameter = self.q(58, 'm')
    rotor_speed = self.q(30.8, 'rpm').to('rad/s')
    wind_speed = self.q(11.5, 'm/s')

    # Calculate blade radius from diameter
    blade_radius = blade_diameter / 2

    # Calculate Tip Speed Ratio (TSR)
    tsr = (rotor_speed * blade_radius) / wind_speed

    q8.printr(tsr)
  
  def seven(self):
    '''
    A wind farm is proposed with 24 wind turbines arranged 
    in a rectangular array: 3 rows with 8 turbines in each row. 
    The turbines are Vestas 825kW turbines with a blade diameter of 52m. 
    Estimate the area of this windfarm. Use 3D between turbines in a row, 
    12D between rows, and 100m setback. Estimate the area of this windfarm.
    '''
    num_rows = 3
    num_turbines_per_row = 8
    blade_diameter = self.q(52, 'm')
    spacing_between_turbines = 3 * blade_diameter
    spacing_between_rows = 12 * blade_diameter
    setback = self.q(100, 'm')

    # Calculate width and length of the wind farm area
    width = (num_turbines_per_row - 1) * spacing_between_turbines
    length = (num_rows - 1) * spacing_between_rows

    # Add setbacks
    total_width = width + 2 * setback
    total_length = length + 2 * setback

    # Calculate total area
    total_area = total_width * total_length

    q8.printr(total_area.to('km**2'))

x = q8()
x.one()
x.two()
x.three()
x.four()
x.five()
x.six()
x.seven()