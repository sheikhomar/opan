import numpy as np
import sympy as sy

from fractions import Fraction
from IPython.display import HTML, display

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Show decimal numbers as fractions
# np.set_printoptions(formatter={'all':lambda x: str(Fraction(x).limit_denominator())})

def load_custom_styles(stylesheet='styles.css'):
    with open(stylesheet) as f:
        css = f.read().replace(';', ' !important;')
    display(HTML('<style type="text/css">%s</style>Stylesheet "%s" loaded.'% (css, stylesheet)))

def to_frac(x):
    return str(Fraction(x).limit_denominator())

def to_decimal(numpy_array):
    return [float(el) for el in numpy_array]

def format_vector(vec):
    output = '['
    for i, entry in enumerate(vec):
        if i > 0:
            output += ', '
        output += '{0: .4f}'.format(float(entry))
    output += ']'
    return output

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
    ax.set_axisbelow(True)
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


def create_simplex_tableau(A, b):
    A = np.array(A)
    b = np.array(b)
    html_out = '<table class="simplex-tableaux"><thead><tr>'
    N_variables = A.shape[1]
    N_slack = A.shape[0]-1
    N_rows = A.shape[0]
    for i in range(N_variables):
        html_out += f'<th style="text-align:center;">$x_{i}$</th>'
    for i in range(N_slack):
        html_out += f'<th style="text-align:center;">$s_{i}$</th>'
    html_out += f'<th style="text-align:center;">$M$</th>'
    html_out += f'<th style="text-align:center;">$b$</th>'
    html_out += '</tr></thead><tbody>'
    for row in range(N_rows):
        is_last_row = row == (N_rows-1)
        html_out += f'<tr class="{"last-row" if is_last_row else ""}">'

        for column in range(N_variables):
            cell = A[row, column]
            min_row_val = A[row].min()
            class_name = 'smallest-value' if is_last_row and min_row_val == cell else ''
            html_out += f'<td class="{class_name}">{cell}</td>'
        for column in range(N_rows):
            html_out += f'<td>{1 if column == row else 0}</td>'
        html_out += f'<td>{b[row]}</td>'
        html_out += '</tr>'
    html_out += ''
    html_out += '</tbody></table>'
    return HTML(html_out)


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

    def array_cell(self, cell_index, val, element_format='%.5f'):
        html = '\\begin{bmatrix}'
        for el in val:
            html += element_format % el
            html += ' \\\\ '
        html += '\\end{bmatrix}'
        html += '&nbsp;'
        self.text_cell(cell_index, html)

    def as_html(self):
        return self._repr_html_()

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
