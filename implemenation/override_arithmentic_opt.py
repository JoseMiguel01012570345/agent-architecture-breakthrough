import math
import sys

max_index = sys.maxsize
epsilon = 1e-9


class Interval:

    def __init__(self, lower, upper):
        if lower > upper:
            return
            raise ValueError(
                "Lower bound must be less than or equal to the upper bound."
            )
        self.lower = lower
        self.upper = upper

    def __add__(self, other):
        """Addition of intervals: [a, b] + [c, d] = [a + c, b + d]"""

        if isinstance(other, Interval):
            return Interval(self.lower + other.lower, self.upper + other.upper)

        return Interval(self.lower + other[0], self.upper + other[1])

    def __sub__(self, other):
        """Subtraction of intervals: [a, b] - [c, d] = [a - d, b - c]"""
        if isinstance(other, Interval):
            return Interval(self.lower - other.upper, self.upper - other.lower)
        return Interval(self.lower - other, self.upper - other)

    def __neg__(self):
        """Negate the current interval."""
        return Interval(-self.upper, -self.lower)

    def __mul__(self, other):
        """Multiplication of intervals: [a, b] * [c, d] = [min(ac, ad, bc, bd), max(ac, ad, bc, bd)]"""
        if isinstance(other, Interval):
            products = [
                self.lower * other.lower,
                self.lower * other.upper,
                self.upper * other.lower,
                self.upper * other.upper,
            ]
            return Interval(min(products), max(products))
        return Interval(self.lower * other, self.upper * other)

    def __truediv__(self, other):
        """Division of intervals: [a, b] / [c, d] = [a, b] * [1/d, 1/c] if 0 ∉ [c, d]"""
        if isinstance(other, Interval):
            if other.lower <= 0 <= other.upper:
                other.lower = epsilon
                # raise ZeroDivisionError(
                #     "Interval division by an interval containing zero is undefined."
                # )
            return self * Interval(1 / other.upper, 1 / other.lower)
        return Interval(self.lower / other, self.upper / other)

    def __str__(self):
        return f"[{self.lower}, {self.upper}]"

    def __pow__(self, k):
        """
        Compute [a, b]^[c, d] for real intervals.
        Returns (min_result, max_result) or raises ValueError for invalid cases.
        """

        a, b = self.lower, self.upper
        c, d = k.lower, k.upper

        if a < 0:
            if b < 0:
                b = epsilon

            b -= a
            a = epsilon

        # Check validity
        if a == 0:
            a = epsilon
            b += a

        # Compute critical points

        candidates = []
        for x in [a, b]:
            for y in [c, d]:
                m = 0
                try:
                    m = math.pow(x, y)
                except:
                    m = max_index

                candidates.append(m)

        return Interval(min(candidates), max(candidates))
