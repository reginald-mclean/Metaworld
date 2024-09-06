import enum
from typing import overload

import numpy as np
import numpy.typing as npt

class SigmoidType(enum.Enum):
    gaussian = 0
    hyperbolic = enum.auto()
    long_tail = enum.auto()
    recriprocal = enum.auto()
    cosine = enum.auto()
    linear = enum.auto()
    quadratic = enum.auto()
    tanh_squared = enum.auto()

DEFAULT_VALUE_AT_MARGIN = 0.1

@overload
def tolerance(
    x: float,
    bounds: tuple[float, float] = (0.0, 0.0),
    margin: float = 0.0,
    sigmoid: SigmoidType = SigmoidType.gaussian,
    value_at_margin: float = DEFAULT_VALUE_AT_MARGIN,
) -> float: ...
@overload
def tolerance(
    x: np.float64,
    bounds: tuple[float, float] = (0.0, 0.0),
    margin: np.float64 = np.float64(0.0),
    sigmoid: SigmoidType = SigmoidType.gaussian,
    value_at_margin: np.float64 = np.float64(DEFAULT_VALUE_AT_MARGIN),
) -> np.float64: ...
def hamacher_product(a: float, b: float) -> float: ...
def rect_prism_tolerance(
    curr: npt.NDArray[np.float64],
    zero: npt.NDArray[np.float64],
    one: npt.NDArray[np.float64],
) -> float: ...
