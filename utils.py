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

class HtmlTableBuilder:
    def __init__(self, column_headers):
        self._column_headers = column_headers
        self._ncols = len(self._column_headers)
        # Create the first row
        self._rows = []
        self._current_row = -1
    
    def new_row(self):
        self._rows.append([''] * self._ncols)
        self._current_row += 1
        
    def text_cell(self, cell_index, val):
        self._rows[self._current_row][cell_index] = val
    
    def math_cell(self, cell_index, val):
        self._rows[self._current_row][cell_index] = '${0}$'.format(val)

    def array_cell(self, cell_index, val, formatting=':.5f'):
        html = '$\\begin{bmatrix}'
        for el in val:
            html += '{0} \\\\'.format(el)
        html += '\\end{bmatrix}$'
        self.text_cell(cell_index, html)
    
    def _repr_html_(self):
        html_out = '<table><thead><tr>'
        for header in self._column_headers:
            html_out += '<th style="text-align:center;">{0}</th>'.format(header)
        html_out += '</tr></thead><tbody>'
        for row in self._rows:
            html_out += '<tr>'
            
            for cell in range(self._ncols):
                if cell < len(row):
                    html_out += '<td style="text-align:left;">{0}</td>'.format(row[cell])
                else:
                    html_out += '<td>.</td>'
            
            html_out += '</tr>'
        html_out += ''
        html_out += '</tbody></table>'
        return html_out
