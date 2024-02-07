import pandas as pd
import pint
import cmath
import math

class q2:
  def __init__(self) -> None:
    self.q = pint.UnitRegistry(autoconvert_offset_to_baseunit = True).Quantity
  
  def one(self):
    '''
    A 1500 MW steam power plant burns 157 kg/s of bituminous coal, 
    which has a heating value of 27,300 kJ/kg. What is the efficiency of this power plant?
    '''
    prod = (self.q(157, 'kg/s') * self.q(27300,'kJ/kg')).to('MW')
    req = self.q(1500, 'MW')
    print(f"Efficiency: {req / prod}")

  def two(self):
    '''
    Calculate the mass of methane contained in a cylinder 
    with a volume of 54liters at 15oC and 170 Bar . Assume ideal gas.
    '''

    # PV=nRT
    v = self.q(54, 'liters')
    p = self.q(170, 'bar')
    t = self.q(15, 'degC')
    r = self.q(8.3144621, 'J/mol/degK')

    n = (p*v/(r*t)).to('mole')
    print(n, 
          (n * self.q(16.04, 'g/mole')).to('kg'))
    
  def three(self):
    '''
    For a balanced, positive sequence system with phase A 
    voltage VA=120<(50)V, what are the other phase voltages?
    '''

    # Given phase A voltage in polar form: magnitude and angle in degrees
    V_A_mag = 120
    V_A_angle_deg = 50

    # Convert phase A voltage to rectangular coordinates (complex number)
    V_A_rect = cmath.rect(V_A_mag, math.radians(V_A_angle_deg))

    # Calculate phase B voltage by rotating phase A voltage by -120 degrees
    V_B_rect = V_A_rect * cmath.exp(-1j * math.radians(120))

    # Calculate phase C voltage by rotating phase A voltage by 120 degrees
    V_C_rect = V_A_rect * cmath.exp(1j * math.radians(120))

    # Convert rectangular coordinates to polar coordinates for phase B and phase C voltages
    V_B_mag, V_B_angle_rad = cmath.polar(V_B_rect)
    V_C_mag, V_C_angle_rad = cmath.polar(V_C_rect)

    # Convert angles from radians to degrees
    V_B_angle_deg = math.degrees(V_B_angle_rad)
    V_C_angle_deg = math.degrees(V_C_angle_rad)

    print("VB =", round(V_B_mag, 2), "<(", round(V_B_angle_deg, 2), ")")
    print("VC =", round(V_C_mag, 2), "<(", round(V_C_angle_deg, 2), ")")

  def four(self):
    '''
    Given a single phase voltage VA=7200<(20)V, what is the line 
    current delivered to a single phase, 750ohm load?
    '''

    # Given phase voltage in polar form: magnitude and angle in degrees
    V_A_mag = 7200
    V_A_angle_deg = 20

    # Convert phase voltage to rectangular coordinates (complex number)
    V_A_rect = cmath.rect(V_A_mag, math.radians(V_A_angle_deg))

    # Convert phase voltage to line voltage for a single-phase system
    V_L_rect = cmath.sqrt(3) * V_A_rect

    # Given load impedance
    Z = 750  # ohms

    # Calculate line current using Ohm's Law
    I_line = abs(V_L_rect) / Z  # Magnitude of the line current

    # Calculate phase current by dividing line current by sqrt(3)
    I_phase = I_line / math.sqrt(3)

    # Calculate the angle of the phase current (same as the phase voltage angle for a balanced system)
    I_phase_angle_deg = V_A_angle_deg

    # Print phase current magnitude and angle
    print("Phase current delivered to the load:", round(I_phase, 2), "Amperes")
    print("Angle of the phase current:", round(I_phase_angle_deg, 2), "degrees")

  def five(self):
    '''
    Line currents of 15<(30)A are supplied to a balanced 3-phase load from a 7200<(0)V source. 
    What is the total 3 phase power from the source?
    '''

    # Given line current magnitude and angle in degrees
    I_line_mag = self.q(15, 'ampere')
    I_line_angle_deg = 30

    # Given line voltage magnitude
    V_line_mag = self.q(7200, 'amperes')

    # Convert angle from degrees to radians
    i_angle_rad = math.radians(I_line_angle_deg)

    # Calculate total three-phase power
    P_total = math.sqrt(3) * V_line_mag * I_line_mag * math.cos(i_angle_rad)

    # Calculate power factor angle
    power_factor_angle_rad = math.acos(math.cos(i_angle_rad))
    power_factor_angle_deg = math.degrees(power_factor_angle_rad)

    # Convert total three-phase power to polar form (magnitude and angle)
    P_total_polar_mag = abs(P_total) / 1000  # Convert from W to kW
    P_total_polar_angle_deg = power_factor_angle_deg  # Since it's a balanced system, all phases have the same power factor angle

    print("Total three-phase power:", P_total_polar_mag, "<", P_total_polar_angle_deg)

  def six(self):
    '''
    Question 7 (10 points) 
    A 30 hp 460V 3-phase motor requires 35 amps at full load. 
    The power factor is 0.86 Determine the efficiency of the motor.
    '''

    # Given values
    horsepower = self.q(30, 'hp')
    line_voltage = self.q(460,'volt')
    line_current = self.q(35, 'ampere')
    power_factor = 0.86

    # Calculate input power (Pin)
    P_in = math.sqrt(3) * line_voltage * line_current * power_factor

    P_out = horsepower

    # Calculate efficiency
    efficiency = (P_out / P_in).to_base_units()

    print(f"Motor Efficiency: {efficiency}")


  def all(self):
    q2.one(self)
    q2.two(self)
    q2.three(self)
    q2.four(self)
    q2.five(self)
    q2.six(self)

q2().all()