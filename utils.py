import numpy as np
from fractions import Fraction

# Show decimal numbers as fractions
np.set_printoptions(formatter={'all':lambda x: str(Fraction(x).limit_denominator())})

def to_frac(x):
    return str(Fraction(x).limit_denominator())

def to_decimal(numpy_array):
    return [float(el) for el in numpy_array]

def format_vector(vec):
    return str(['{0:0.4f}'.format(float(entry)) for entry in vec])
