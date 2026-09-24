# -*- coding: utf-8 -*-
"""Interpreter-independent float accumulation.

CPython 3.12 changed the algorithm behind the built-in ``sum()`` for floats
(compensated summation), so a pipeline that averages metrics with ``sum()``
produces different last-place bits on 3.9-3.11 than on 3.12+. This module pins
the accumulation to one pure-Python implementation: the same Kahan-Babuska-
Neumaier scheme, written in Python, so every reported figure is bit-stable on
any interpreter the bundle is run under.

Use :func:`csum` where the addends are floats (metric values, bootstrap
draws). Plain integer counts stay on the built-in ``sum()``, which is exact.
"""
from math import fabs, isfinite

__all__ = ['csum']


def csum(values, start=0.0):
    """Compensated sum of an iterable of floats (deterministic across CPython)."""
    partial = float(start)
    comp = 0.0
    for x in values:
        x = float(x)
        if not isfinite(x):
            # Mirrors the built-in's fallback: no compensation once the running
            # total leaves the finite range.
            partial += x
            comp = 0.0
            continue
        t = partial + x
        if fabs(partial) >= fabs(x):
            comp += (partial - t) + x
        else:
            comp += (x - t) + partial
        partial = t
    return partial + comp
