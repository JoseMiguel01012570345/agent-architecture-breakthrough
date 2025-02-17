import math
import numpy as np
import os

max_index = np.finfo(np.float32).max
epsilon = 1e-9


class Interval:

    lower = 0
    upper = 0

    def __init__(self, lower, upper):
        if lower > upper:
            return
            raise ValueError(
                "Lower bound must be less than or equal to the upper bound."
            )
        self.lower = lower
        self.upper = upper

    def check_interval_consistency(self, interval):

        if interval.lower > interval.upper:
            return
            raise ValueError(
                "Lower bound must be less than or equal to the upper bound."
            )
        return interval

    def __add__(self, other):
        """Addition of intervals: [a, b] + [c, d] = [a + c, b + d]"""

        if isinstance(other, Interval):
            lower = self.watch_overflow(self.lower + other.lower)
            upper = self.watch_overflow(self.upper + other.upper)

            return self.check_interval_consistency(Interval(lower=lower, upper=upper))

        lower = self.watch_overflow(self.lower + other)
        upper = self.watch_overflow(self.upper + other)

        return self.check_interval_consistency(Interval(lower=lower, upper=upper))

    def __sub__(self, other):
        """Subtraction of intervals: [a, b] - [c, d] = [a - d, b - c]"""
        if isinstance(other, Interval):
            lower = self.watch_overflow(self.lower - other.upper)
            upper = self.watch_overflow(self.upper - other.lower)

            return self.check_interval_consistency(Interval(lower=lower, upper=upper))

        lower = self.watch_overflow(self.lower - other)
        upper = self.watch_overflow(self.upper - other)
        return self.check_interval_consistency(Interval(lower=lower, upper=upper))

    def __neg__(self):
        """Negate the current interval."""
        lower = self.watch_overflow(min(-self.upper, -self.lower))
        upper = self.watch_overflow(max(-self.lower, -self.upper))
        return self.check_interval_consistency(Interval(lower=lower, upper=upper))

    def __mul__(self, other):
        """Multiplication of intervals: [a, b] * [c, d] = [min(ac, ad, bc, bd), max(ac, ad, bc, bd)]"""
        if isinstance(other, Interval):
            products = [
                self.lower * other.lower,
                self.lower * other.upper,
                self.upper * other.lower,
                self.upper * other.upper,
            ]
            lower = self.watch_overflow(min(products))
            upper = self.watch_overflow(max(products))
            return self.check_interval_consistency(Interval(lower=lower, upper=upper))

        lower = self.watch_overflow(self.lower * other)
        upper = self.watch_overflow(self.upper * other)
        return self.check_interval_consistency(Interval(lower=lower, upper=upper))

    def __truediv__(self, other):
        """Division of intervals: [a, b] / [c, d] = [a, b] * [1/d, 1/c] if 0 ∉ [c, d]"""
        if isinstance(other, Interval):
            if other.lower <= 0 <= other.upper:
                if abs(other.lower) > other.upper:
                    other.upper = -epsilon
                else:
                    other.lower = epsilon
            div = [
                self.lower / other.lower,
                self.lower / other.upper,
                self.upper / other.lower,
                self.upper / other.upper,
            ]
            self_interval = Interval(lower=min(div), upper=max(div))

            return self.check_interval_consistency(interval=self_interval)

        lower = self.watch_overflow(self.lower / other)
        upper = self.watch_overflow(self.upper / other)

        return self.check_interval_consistency(Interval(lower=lower, upper=upper))

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

        lower = self.watch_overflow(min(candidates))
        upper = self.watch_overflow(max(candidates))

        return self.check_interval_consistency(Interval(lower=lower, upper=upper))

    def watch_overflow(self, x):

        x = np.float32(x)

        if np.isposinf(x):
            return max_index
        if np.isneginf(x):
            return -max_index

        return x
