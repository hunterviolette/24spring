import pint

class UnitConversion:

  def __init__(self) -> None:
    uReg = pint.UnitRegistry(autoconvert_offset_to_baseunit = True)
    uReg.define('mtpd = metric_ton / day')
    uReg.define('million_BTU = 1e6 * BTU')
    uReg.define('dollar = [currency] = $')

    self.q, self.ureg = uReg.Quantity, uReg

  @staticmethod
  def LimitingReagent(uo):
    if len(uo["reaction"]["reagents"].keys()) == 1: 
      return next(iter(uo["reaction"]["reagents"]))
    
    else: return uo["limiting_reagent"]

  def debug(self):
    power = (self.q(2.4465*3, 'kmol/h') 
            * 2 * self.q(1,'faraday_constant')
            * self.q(3,'V')
            ).to('kW')

    print(power)

if __name__ == "__main__":
  UnitConversion().debug()