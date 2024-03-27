import pandas as pd
import pint
import numpy_financial as npf

from chempy import Substance


class q7:
  def __init__(self) -> None:
    uReg = pint.UnitRegistry(autoconvert_offset_to_baseunit = True)
    uReg.define('dollars = [currency]')
    self.q=uReg.Quantity

    self.g = self.q(9.81, 'm/s**2')
    self.h2mass = self.q(Substance.from_formula('H2').mass, 'gram/mol')
  
  @staticmethod
  def printr(x, rounding: int = 2):
    print(round(x,rounding))

  def one(self):
    '''
    A high-efficiency 200 hp motor costs $2027 more than a standard motor, 
    but saves $516/year in electricity costs. Determine the simple payback for this motor.
    '''
    cost = self.q(2027, 'dollars')
    savings = self.q(516, 'dollars/year')

    q7.printr(cost / savings)

  def two(self):
    '''
    A person puts $4000 into an investment that pays 6% interest, 
    compounded annually. What is the value of the investment in 15 years?
    '''
    principal = self.q(4000, 'dollars')
    rate = 0.06
    years = 15

    q7.printr(principal * (1 + rate) ** years)
  
  def three(self):
    '''
    What is the present value of two cash payments: 
    $3000 one year from now, and $4000 3 years from now. Use a discount rate of 5%.
    '''
    payment1 = self.q(3000, 'dollars')
    payment2 = self.q(4000, 'dollars')

    discount_rate = 0.05
    present_value = payment1 / (1 + discount_rate) + payment2 / (1 + discount_rate) ** 3

    q7.printr(present_value)
  
  def four(self):
    '''
    A $2000 investment pays $1000 at the end of year 1, 
    $1500 at the end of year 2, and $2500 at the end of year 3. 
    Using a discount rate of 7%, what is the net present value of this investment?
    '''
    cashflows = [-2000, 1000, 1500, 2500]  # Cashflows at each period
    discount_rate = 0.07  # Discount rate

    npv = sum(cf / (1 + discount_rate) ** t for t, cf in enumerate(cashflows))
    q7.printr(npv)
  
  def five(self):
    '''
    A high-efficiency air conditioner costs $5000 and has a life of 6 years. 
    Assuming an interest rate of 10%, what annual energy cost savings 
    will be necessary to justify this expense?
    '''
    cost = self.q(5000, 'dollars')
    life_years = 6
    discount_rate = 0.10

    savings_needed = cost / ((1 - (1 + discount_rate) ** -life_years) / discount_rate)
    q7.printr(savings_needed)
  
  def six(self):
    '''
    A heat pump costs $8000 more than an electric resistance heater, 
    but is expected to save $1500/year over 10 years. 
    What is the net present worth of the heat pump, assuming a discount rate of 9%?
    '''
    cost_difference = self.q(8000, 'dollars')
    annual_savings = self.q(1500, 'dollars')
    life_years = self.q(10, 'years').magnitude
    discount_rate = 0.09

    cashflows = [-cost_difference] + [annual_savings] * life_years
    npv = sum(cf / (1 + discount_rate) ** t for t, cf in enumerate(cashflows))
    q7.printr(npv)

  def seven(self):
    '''
    You invest $10,000 and after 8 years your investment is worth $18,000.
    What is the rate of return on your investment?
    '''
    initial_investment = self.q(10000, 'dollars')
    final_value = self.q(18000, 'dollars')
    years = 8

    rate_of_return = ((final_value / initial_investment) ** (1 / years)) - 1
    q7.printr(rate_of_return.magnitude * 100)

  def eight(self):
    '''
    A company invests $6400 in an efficient lighting system that is expected to save $2000/year for 5 years. 
    What is the internal rate of return (IRR) on this investment?
    '''
    initial_investment = 6400  # in dollars
    annual_savings = 2000  # in dollars/year
    life_years = 5

    # Cashflows including the initial investment and the annual savings
    cashflows = [-initial_investment] + [annual_savings] * life_years

    # Calculate IRR
    irr = npf.irr(cashflows)
    q7.printr(irr * 100)

  def nine(self):
    '''  
    A 3 ton ground source heat pump has an installed cost of $22,550. 
    This cost is to be financed by a 25 year loan at 8% interest. 
    The annual payments on this loan are
    '''
    q7.printr(
          npf.pmt(
              rate=0.08, 
              nper=25, 
              pv=22550
            )
          )

  def ten(self):
    '''
    A 30 kW photovoltaic system has an installation cost of $114,600. 
    There are no annual costs other than the payments on a 7% 20-year loan. 
    If the photovoltaic system has a capacity factor of 0.22, 
    determine the cost of the electricity generated.
    '''
    annual_payment = npf.pmt(rate=.07, nper=20, pv=114600)

    system_capacity = self.q(30,'kW/h')
    capacity_factor = 0.22

    q7.printr(
          self.q(annual_payment, 'dollars') / 
          (system_capacity * capacity_factor).to('kW/year')
          )
    
  def eleven(self):
    '''
    A 35 MW biomass power plant costs $130 million to construct. 
    The plant has a capacity factor of 0.80. 
    Using an interest rate of 10% and a project life of 28 years, 
    calculate the levelized fixed cost. 
    (Assume that the fixed charge rate is equal to the capital recovery factor.)
    '''
    capacity = self.q(35, 'MW/h')
    pv = self.q(130e6, 'dollars')  # $130 million
    n = self.q(28,'years') 
    capacity_factor = 0.80

    annual_payment = npf.pmt(rate=.1, nper=n.magnitude, pv=-pv.magnitude)
    annual_production = (capacity_factor * capacity).to('kW/year')

    levelized_cost = self.q(annual_payment, 'dollars') / (annual_production)

    q7.printr(levelized_cost)

  def twelve(self):
    '''
    The initial annual cost (fuel and O&M) for a gas turbine engine is $0.049/kWh. 
    Using a discount rate of 10% and a cost escalation rate of 4%/yr over a 20 year lifetime, 
    calculate the levelized annual cost.
    '''
    cost = .049
    escalation_rate = 0.04 

    annual_payment = npf.pmt(rate=.1, nper=20, pv=-cost)
    levelized_annual_cost = cost + (escalation_rate * cost) + annual_payment

    q7.printr(levelized_annual_cost, 4)

x = q7()
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
x.twelve()