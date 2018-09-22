import numpy as np
import matplotlib.pyplot as plt
import sympy as sy

from fractions import Fraction
from IPython.display import HTML, display

# Show decimal numbers as fractions
np.set_printoptions(formatter={'all':lambda x: str(Fraction(x).limit_denominator())})

def to_frac(x):
    return str(Fraction(x).limit_denominator())

def to_decimal(numpy_array):
    return [float(el) for el in numpy_array]

def format_vector(vec):
    return str(['{0:0.4f}'.format(float(entry)) for entry in vec])

def prepare_plot(x, y, xlimit=(-10, 10), ylimit=None, show_spines=True, figsize=(10, 6)):
    if ylimit is None:
        y_high = np.max(y)
        y_low = np.min(y)
        space = (y_high - y_low) / 15
        ylimit = (y_low - space, y_high + space)
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$f(x)$')
    ax.set_xlim(*xlimit)
    ax.set_ylim(*ylimit)
    if show_spines:
        x_min = min(xlimit[0], xlimit[1])
        if xlimit[0] < 0 and xlimit[1] > 0:
            x_min = 0.0
        y_min = min(ylimit[0], ylimit[1])
        if ylimit[0] < 0 and ylimit[1] > 0:
            y_min = 0.0
        ax.spines['left'].set_position(('data', x_min))
        ax.spines['bottom'].set_position(('data', y_min))
        ax.spines['left'].set_smart_bounds(True)
        ax.spines['bottom'].set_smart_bounds(True)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
    return fig, ax

