import numpy as np
import math
from override_arithmentic_opt import Interval
import sys

max_index = np.finfo(np.float32).max
epsilon = 1e-9
overflow_value = 610

output_intervalar_functions_avaliable = [  # output agents must cover all of the y_true values of dataset
    (
        lambda x: Interval(lower=x.lower, upper=x.upper),
        lambda y: Interval(lower=y.lower, upper=y.upper),
    ),
    (
        lambda x: Interval(2, 2) * x,
        lambda y: y / Interval(2, 2),
    ),
    # (
    #     lambda x: Interval(3, 3) * x + Interval(1, 1),
    #     lambda y: (y - Interval(1, 1)) / Interval(3, 3),
    # ),
    (
        lambda x: x ** Interval(3, 3),
        lambda y: y ** (Interval(1, 1) / Interval(3, 3)),
    ),
    # (
    #     lambda x: Interval(math.e, math.e) ** x,
    #     lambda y: log(a=check_overflow(y.lower), b=check_overflow(y.upper)),
    # ),
    # (
    #     lambda x: log(a=check_overflow(x.lower), b=check_overflow(x.upper)),
    #     lambda y: Interval(math.e, math.e)
    #     ** Interval(lower=y.lower, upper=y.upper),
    # ),
    # (
    #     lambda x: sin(check_overflow(x.lower), check_overflow(x.upper)),
    #     lambda y: (
    #         interval_arcsin(check_overflow(y.lower), check_overflow(y.upper))
    #     ),
    # ),
    (
        lambda x: interval_tan(check_overflow(x.lower), check_overflow(x.upper)),
        lambda y: Interval(
            math.atan(check_overflow(y.lower)), math.atan(check_overflow(y.upper))
        ),
    ),
    (
        lambda x: (
            Interval(1, 1) / x
            if check_overflow(x.lower) > 0 or 0 > check_overflow(x.upper)
            else Interval(max_index, max_index)
        ),
        lambda y: (
            Interval(1, 1) / y
            if y.lower > 0 or 0 > y.upper
            else Interval(max_index, max_index)
        ),
    ),
    # (
    #     lambda x: interval_cos(check_overflow(x.lower), check_overflow(x.upper)),
    #     lambda y: (
    #         interval_arccos(check_overflow(y.lower), check_overflow(y.upper))
    #     ),
    # ),
    (
        lambda x: Interval(1, 1) / (Interval(1, 1) + Interval(math.e, math.e) ** (-x)),
        lambda y: special_log(a=check_overflow(y.lower), b=check_overflow(y.upper)),
    ),
    # (
    #     lambda x: Interval(
    #         math.sinh(check_overflow(check_overflow(x.upper))),
    #         math.sinh(check_overflow(check_overflow(x.upper))),
    #     ),
    #     lambda y: interval_arcsinh(check_overflow(y.lower), check_overflow(y.upper)),
    # ),
    # (
    #     lambda x: interval_cosh(
    #         check_overflow(x.lower),
    #         check_overflow(x.upper),
    #     ),
    #     lambda y: (interval_arccosh(check_overflow(y.lower), check_overflow(y.upper))),
    # ),
    (
        lambda x: Interval(
            math.atan(check_overflow(x.lower)),
            math.atan(check_overflow(x.upper)),
        ),
        lambda y: interval_tan(check_overflow(y.lower), check_overflow(y.upper)),
    ),
    # (
    #     lambda x: special_log(a=check_overflow(x.lower), b=check_overflow(x.upper)),
    #     lambda y: Interval(1, 1)
    #     / (
    #         Interval(1, 1)
    #         + Interval(math.e, math.e) ** (-Interval(lower=y.lower, upper=y.upper))
    #     ),
    # ),
    (
        lambda x: x ** Interval(1, 1) / Interval(3, 3),
        lambda y: y ** Interval(3, 3),
    ),
    (
        lambda x: Interval(1, 1) - Interval(lower=x.lower, upper=x.upper),
        lambda y: Interval(1, 1) - Interval(lower=y.lower, upper=y.upper),
    ),
    # (
    #     lambda x: log10(a=check_overflow(x.lower), b=check_overflow(x.upper)),
    #     lambda y: Interval(10, 10) ** y,
    # ),
]


intervalar_functions_avaliable = [
    (
        lambda x: Interval(lower=x.lower, upper=x.upper),
        lambda y: Interval(lower=y.lower, upper=y.upper),
    ),
    # (
    #     lambda x: Interval(2, 2) * x,
    #     lambda y: y / Interval(2, 2),
    # ),
    # (
    #     lambda x: Interval(3, 3) * x + Interval(1, 1),
    #     lambda y: (y - Interval(1, 1)) / Interval(3, 3),
    # ),
    # (
    #     lambda x: x ** Interval(3, 3),
    #     lambda y: y ** (Interval(1, 1) / Interval(3, 3)),
    # ),
    # (
    #     lambda x: Interval(math.e, math.e) ** x,
    #     lambda y: log(a=check_overflow(y.lower), b=check_overflow(y.upper)),
    # ),
    (
        lambda x: log(a=check_overflow(x.lower), b=check_overflow(x.upper)),
        lambda y: Interval(math.e, math.e) ** Interval(lower=y.lower, upper=y.upper),
    ),
    (
        lambda x: sin(check_overflow(x.lower), check_overflow(x.upper)),
        lambda y: (interval_arcsin(check_overflow(y.lower), check_overflow(y.upper))),
    ),
    # (
    #     lambda x: interval_tan(check_overflow(x.lower), check_overflow(x.upper)),
    #     lambda y: Interval(
    #         math.atan(check_overflow(y.lower)), math.atan(check_overflow(y.upper))
    #     ),
    # ),
    # (
    #     lambda x: (
    #         Interval(1, 1) / x
    #         if check_overflow(x.lower) > 0 or 0 > check_overflow(x.upper)
    #         else Interval(max_index, max_index)
    #     ),
    #     lambda y: (
    #         Interval(1, 1) / y
    #         if y.lower > 0 or 0 > y.upper
    #         else Interval(max_index, max_index)
    #     ),
    # ),
    (
        lambda x: interval_cos(check_overflow(x.lower), check_overflow(x.upper)),
        lambda y: (interval_arccos(check_overflow(y.lower), check_overflow(y.upper))),
    ),
    # (
    #     lambda x: Interval(1, 1) / (Interval(1, 1) + Interval(math.e, math.e) ** (-x)),
    #     lambda y: special_log(a=check_overflow(y.lower), b=check_overflow(y.upper)),
    # ),
    # (
    #     lambda x: Interval(
    #         math.sinh(check_overflow(check_overflow(x.upper))),
    #         math.sinh(check_overflow(check_overflow(x.upper))),
    #     ),
    #     lambda y: interval_arcsinh(check_overflow(y.lower), check_overflow(y.upper)),
    # ),
    # (
    #     lambda x: interval_cosh(
    #         check_overflow(x.lower),
    #         check_overflow(x.upper),
    #     ),
    #     lambda y: (interval_arccosh(check_overflow(y.lower), check_overflow(y.upper))),
    # ),
    (
        lambda x: Interval(
            math.atan(check_overflow(x.lower)),
            math.atan(check_overflow(x.upper)),
        ),
        lambda y: interval_tan(check_overflow(y.lower), check_overflow(y.upper)),
    ),
    (
        lambda x: special_log(a=check_overflow(x.lower), b=check_overflow(x.upper)),
        lambda y: Interval(1, 1)
        / (
            Interval(1, 1)
            + Interval(math.e, math.e) ** (-Interval(lower=y.lower, upper=y.upper))
        ),
    ),
    # (
    #     lambda x: x ** Interval(1, 1) / Interval(3, 3),
    #     lambda y: y ** Interval(3, 3),
    # ),
    (
        lambda x: Interval(1, 1) - Interval(lower=x.lower, upper=x.upper),
        lambda y: Interval(1, 1) - Interval(lower=y.lower, upper=y.upper),
    ),
    # (
    #     lambda x: log10(a=check_overflow(x.lower), b=check_overflow(x.upper)),
    #     lambda y: Interval(10, 10) ** y,
    # ),
]


def log(a, b):
    a, b = convert_to_max_integer(a=a, b=b)
    if a < 0:
        a, b = abs(a) + epsilon, abs(a) + epsilon

    return Interval(np.log(a), np.log(b))


def special_log(a, b):
    a, b = convert_to_max_integer(a=a, b=b)
    if not 0 < a < 1:
        a = epsilon

    if not 0 < b < 1:
        b = 1 - epsilon

    return Interval(np.log(a / (1 - a)), np.log(b / (1 - b)))


def log10(a, b):
    a, b = convert_to_max_integer(a=a, b=b)
    if a > 0:
        a, b = abs(a) + epsilon, abs(a) + epsilon

    try:
        ret = Interval(np.log10(a), np.log10(b))
    except:
        print(a, b)
        ValueError("bla bla bla")

    return ret


def sin(a, b):
    """
    Compute the interval resulting from applying the sine function to the interval [a, b].
    """
    a, b = convert_to_max_integer(a=a, b=b)
    # If the interval spans at least a full period (2π), return [-1, 1]
    if b - a >= 2 * math.pi:
        return Interval(-1.0, 1.0)

    # Find all critical points (where sin(x) = ±1) within [a, b]
    critical_points = []
    # Critical points occur at x = π/2 + kπ for integers k
    k_min = math.ceil((a - math.pi / 2) / math.pi)
    k_max = math.floor((b - math.pi / 2) / math.pi)

    for k in range(k_min, k_max + 1):
        cp = math.pi / 2 + k * math.pi
        if a <= cp <= b:
            critical_points.append(cp)

        if len(critical_points) == 2:
            return Interval(-1, 1)

    # Evaluate sin at endpoints and critical points
    points = [a, b] + critical_points
    sin_values = [math.sin(p) for p in points]
    return Interval(min(sin_values), max(sin_values))


def interval_cosh(a, b):
    """
    Compute the interval of cosh(x) for an interval [a, b].

    Parameters:
    x (tuple): Interval (a, b) where a <= b.

    Returns:
    tuple: Resulting interval (lower, upper).

    Raises:
    ValueError: If input bounds are invalid (a > b).
    """
    a, b = convert_to_max_integer(a=a, b=b)
    if a > b:
        raise ValueError("Invalid interval: lower bound > upper bound")

    contains_zero = a <= 0 <= b
    max_abs = max(abs(a), abs(b))

    if contains_zero:
        lower = 1.0
    else:
        min_abs = min(abs(a), abs(b))
        lower = math.cosh(check_overflow(min_abs))

    upper = math.cosh(check_overflow(max_abs))
    return Interval(min(lower, upper), max(lower, upper))


def interval_tan(a, b):
    """
    Compute the tangent of an interval [a, b].

    Parameters:
    x (tuple): A tuple representing the interval (a, b), where a <= b.

    Returns:
    tuple: The resulting interval (lower, upper).

    Raises:
    ValueError: If the interval contains a vertical asymptote of tan(x) or if a > b.
    """
    a, b = convert_to_max_integer(a=a, b=b)
    if a > b:
        raise ValueError("Invalid interval: lower bound > upper bound")

    # Check if the interval contains any vertical asymptotes of tan(x)
    # Asymptotes occur at x = π/2 + kπ for integers k

    # Find the smallest k such that π/2 + kπ >= a
    k_min = math.ceil((a - math.pi / 2) / math.pi)
    x_k_min = math.pi / 2 + k_min * math.pi

    if x_k_min < b:
        if abs(x_k_min - b) > abs(x_k_min - a):
            a = x_k_min + epsilon
        else:
            b = x_k_min - epsilon

    # Check if the upper bound is exactly an asymptote

    if abs((b - math.pi / 2) / math.pi) < epsilon:
        b += epsilon

    # Compute the tangent values
    lower = math.tan(a)
    upper = math.tan(b)

    return Interval(min(upper, lower), max(lower, upper))


def interval_arcsinh(a, b):
    """
    Compute the inverse hyperbolic sine (arcsinh) of an interval [a, b].

    Parameters:
    x (tuple): A tuple representing the interval (a, b), where a <= b.

    Returns:
    tuple: The resulting interval (lower, upper).

    Raises:
    ValueError: If a > b.
    """
    a, b = convert_to_max_integer(a=a, b=b)
    return Interval(math.asinh(a), math.asinh(b))


def interval_arccosh(a, b):
    """
    Compute the inverse hyperbolic cosine (arccosh) of an interval [a, b].

    Parameters:
    x (tuple): A tuple representing the interval (a, b), where a <= b.

    Returns:
    tuple: The resulting interval (lower, upper).

    Raises:
    ValueError: If a > b or the interval contains values < 1.
    """
    a, b = convert_to_max_integer(a=a, b=b)
    if b < 1:
        b = 1 + epsilon
    if a > b:
        a = b - epsilon
    if a < 1:
        a = 1
    return Interval(math.acosh(a), math.acosh(b))


def interval_arcsin(a, b):
    """
    Compute the inverse sine (arcsin) of an interval [a, b].

    Parameters:
    x (tuple): A tuple representing the interval (a, b), where a <= b.

    Returns:
    tuple: The resulting interval (lower, upper).

    Raises:
    ValueError: If a > b or the interval is outside [-1, 1].
    """

    a, b = convert_to_max_integer(a=a, b=b)
    a, b = np.clip(np.array([a, b]), -1, 1)

    return Interval(math.asin(a), math.asin(b))


def interval_arccos(a, b):
    """
    Compute the inverse sine (arcsin) of an interval [a, b].

    Parameters:
    x (tuple): A tuple representing the interval (a, b), where a <= b.

    Returns:
    tuple: The resulting interval (lower, upper).

    Raises:
    ValueError: If a > b or the interval is outside [-1, 1].
    """

    a, b = convert_to_max_integer(a=a, b=b)
    a, b = np.clip(np.array([a, b]), -1, 1)

    return Interval(math.acos(a), math.acos(b))


def interval_cos(a, b):
    """
    Compute the cosine of an interval [a, b].

    Parameters:
    x (tuple): A tuple representing the interval (a, b), where a <= b.

    Returns:
    tuple: The resulting interval (lower, upper), capturing the full range of cos(x).

    Raises:
    ValueError: If a > b.
    """

    a, b = convert_to_max_integer(a=a, b=b)

    k_start = math.ceil(a / math.pi)
    k_end = math.floor(b / math.pi)

    # Generate critical points (multiples of π within [a, b])
    critical_points = []
    for k in range(k_start, k_end + 1):
        critical_points.append(k * math.pi)
        if len(critical_points) == 2:
            return Interval(-1, 1)

    # Evaluate cos at endpoints and critical points
    points = [a, b] + critical_points
    values = [math.cos(p) for p in points]

    return Interval(min(values), max(values))


def convert_to_max_integer(a, b):
    if np.isposinf(a):
        a = max_index
    if np.isposinf(b):
        b = max_index

    return a, b


def check_overflow(a):

    r = a if a < overflow_value else overflow_value
    r = r if r > -overflow_value else -overflow_value

    return r
