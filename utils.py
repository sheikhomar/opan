import fractions

from fractions import Fraction

def to_frac(x):
    return str(Fraction(x).limit_denominator())