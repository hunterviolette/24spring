import pint

class UnitConversion:

  def __init__(self) -> None:
    uReg = pint.UnitRegistry(autoconvert_offset_to_baseunit = True)
    uReg.define('mtpd = metric_ton / day')
    uReg.define('batch = 1 / (20*min)')
    uReg.define('million_BTU = 1e6 * BTU')
    uReg.define('dollar = [currency] = $')

    self.q = uReg.Quantity