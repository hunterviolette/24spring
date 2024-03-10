
if __name__ == "__main__":
  from unit_registry import UnitConversion
else:
  from src.unit_registry import UnitConversion

class Therm(UnitConversion):

  def __init__(self) -> None:
    super().__init__()

    print(self.q(5, 'm'))

if __name__ == "__main__":
  Therm()