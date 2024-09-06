from libc.math cimport sqrt, log, exp, M_PI, tanh, cos, cosh, acos, acosh, atanh
cimport cython


DEF DEFAULT_VALUE_AT_MARGIN = 0.1


cpdef enum SigmoidType:
    gaussian,
    hyperbolic,
    long_tail,
    recriprocal,
    cosine,
    linear,
    quadratic,
    tanh_squared


def sigmoids(double x, double value_at_1, SigmoidType sigmoid) -> double:
    return _sigmoids(x, value_at_1, sigmoid)


@cython.cdivision(True)
def tolerance(
    double x,
    (double, double) bounds = (0.0, 0.0),
    double margin = 0.0,
    SigmoidType sigmoid = SigmoidType.gaussian,
    double value_at_margin = DEFAULT_VALUE_AT_MARGIN,
) -> double:
    """Returns 1 when `x` falls inside the bounds, between 0 and 1 otherwise.

    Args:
        x: The input.
        bounds: A tuple of floats specifying inclusive `(lower, upper)` bounds for
        the target interval. These can be infinite if the interval is unbounded
        at one or both ends, or they can be equal to one another if the target
        value is exact.
        margin: Parameter that controls how steeply the output decreases as
        `x` moves out-of-bounds.
        * If `margin == 0` then the output will be 0 for all values of `x`
            outside of `bounds`.
        * If `margin > 0` then the output will decrease sigmoidally with
            increasing distance from the nearest bound.
        sigmoid: Choice of sigmoid type. Valid values are 'gaussian', 'hyperbolic',
        'long_tail', 'reciprocal', 'cosine', 'linear', 'quadratic', 'tanh_squared'.
        value_at_margin: A value between 0 and 1 specifying the output when
        the distance from `x` to the nearest bound is equal to `margin`. Ignored
        if `margin == 0`.

    Returns:
        A float or numpy array with values between 0.0 and 1.0.

    Raises:
        ValueError: If `bounds[0] > bounds[1]`.
        ValueError: If `margin` is negative.
    """
    cdef double value, d

    if bounds[0] > bounds[1]:
        raise ValueError("Lower bound must be <= upper bound.")
    if margin < 0:
        raise ValueError(f"`margin` must be non-negative. Current value: {margin}")

    if margin == 0:
        value = 1.0 if bounds[0] <= x and x <= bounds[1] else 0.0
    else:
        d = (bounds[0] - x if x < bounds[1] else x - bounds[1]) / margin
        value = 1.0 if bounds[0] <= x and x <= bounds[1] else _sigmoids(d, value_at_margin, sigmoid)

    return value 

@cython.cdivision(True)
def hamacher_product(double a, double b) -> double:
    """Returns the hamacher (t-norm) product of a and b.

    Computes (a * b) / ((a + b) - (a * b)).

    Args:
        a: 1st term of the hamacher product.
        b: 2nd term of the hamacher product.

    Returns:
        The hammacher product of a and b

    Raises:
        ValueError: a and b must range between 0 and 1
    """
    if not ((0.0 <= a <= 1.0) and (0.0 <= b <= 1.0)):
        raise ValueError("a and b must range between 0 and 1")

    cdef double denominator, h_prod

    denominator = a + b - (a * b)
    h_prod = 0.0
    if denominator > 0:
        h_prod = ((a * b) / denominator)

    if not 0.0 <= h_prod <= 1.0:
        raise AssertionError(f"Invalid h_prod value produced: {h_prod!r}")
    return h_prod

@cython.cdivision(True)
cdef double _sigmoids(double x, double value_at_1, SigmoidType sigmoid):
    if sigmoid == SigmoidType.cosine or sigmoid == SigmoidType.linear or sigmoid == SigmoidType.quadratic:
        if not 0 <= value_at_1 < 1:
            raise ValueError(
                f"`value_at_1` must be nonnegative and smaller than 1, got {value_at_1}."
            )
    else:
        if not 0 < value_at_1 < 1:
            raise ValueError(
                f"`value_at_1` must be strictly between 0 and 1, got {value_at_1}."
            )

    cdef double scale, scaled_x

    if sigmoid == SigmoidType.gaussian:
        scale = sqrt(-2 * log(value_at_1))
        return exp(-0.5 * (x * scale) ** 2)

    elif sigmoid == SigmoidType.hyperbolic:
        scale = acosh(1 / value_at_1)
        return 1 / cosh(x * scale)

    elif sigmoid == SigmoidType.long_tail:
        scale = sqrt(1 / value_at_1 - 1)
        return 1 / ((x * scale) ** 2 + 1)

    elif sigmoid == SigmoidType.recriprocal:
        scale = 1 / value_at_1 - 1
        return 1 / (abs(x) * scale + 1)

    elif sigmoid == SigmoidType.cosine:
        scale = acos(2 * value_at_1 - 1) / M_PI
        scaled_x = x * scale

        if abs(scaled_x) < 1:
            return (1 + cos(M_PI * scaled_x)) / 2
        else:
            return 0.0

    elif sigmoid == SigmoidType.linear:
        scale = 1 - value_at_1
        scaled_x = x * scale
        if abs(scaled_x) < 1:
            return 1 - scaled_x
        else:
            return 0.0

    elif sigmoid == SigmoidType.quadratic:
        scale = sqrt(1 - value_at_1)
        scaled_x = x * scale
        if abs(scaled_x) < 1:
            return 1 - scaled_x**2
        else:
            return 0.0

    elif sigmoid == SigmoidType.tanh_squared:
        scale = atanh(sqrt(1 - value_at_1))
        return 1 - tanh(x * scale) ** 2

    else:
        raise ValueError(f"Invalid sigmoid type {sigmoid!r}.")


cdef bint _in_range(double a, double b, double c):
    return b <= a <= c if c >= b else c <= a <= b

def rect_prism_tolerance(
    double[:] curr,
    double[:] zero,
    double[:] one,
) -> double:
    """Computes a reward if curr is inside a rectangular prism region.

    All inputs are 3D points with shape (3,).

    Args:
        curr: The point that the prism reward region is being applied for.
        zero: The diagonal opposite corner of the prism with reward 0.
        one: The corner of the prism with reward 1.

    Returns:
        A reward if curr is inside the prism, 1.0 otherwise.
    """

    cdef double x_scale, y_scale, z_scale
    cdef double[3] diff

    if _in_range(curr[0], zero[0], one[0]) and _in_range(curr[1], zero[1], one[1]) and _in_range(curr[2], zero[2], one[2]):
        for i in range(3):
            diff[i] = one[i] - zero[i]
        x_scale = (curr[0] - zero[0]) / diff[0]
        y_scale = (curr[1] - zero[1]) / diff[1]
        z_scale = (curr[2] - zero[2]) / diff[2]
        return x_scale * y_scale * z_scale
    else:
        return 1.0
