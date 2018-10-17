import numpy as np
import matplotlib.pyplot as plt
import sympy as sy
import utils as utils
from utils import HtmlTableBuilder

from func import Func

from IPython.display import HTML, display


class ConjugateGradient:
    def __init__(self, f: Func, Q):
        self._f = f
        self._Q = sy.Matrix(Q)

    def find_minimum(self, starting_point):
        x_k = sy.Matrix(starting_point)
        Q = self._Q
        zero_vector = sy.zeros(Q.shape[0], 1);
        max_iterations = Q.shape[0]
        table_headers = ['$k$', '$d^{(k)}$', '$\\alpha_k$', '$x^{(k+1)}$', '$g^{(k+1)}$', '$\\beta_k$']
        table = HtmlTableBuilder(table_headers)
        for k in range(max_iterations):
            table.new_row()
            if k == 0:
                g_k = self._f.gradient_at(x_k)
                if g_k == zero_vector:
                    print('Stopping since g(0) = 0')
                    break
                d_k = -g_k
            else:
                d_k = -g_k + beta_k * d_k

            alpha_k = (-(g_k.T * d_k)/(d_k.T * Q * d_k))[0]
            x_kp1 = x_k + (alpha_k * d_k)
            g_kp1 = self._f.gradient_at(x_kp1)
            beta_k = ((g_kp1.T * Q * d_k)/(d_k.T * Q * d_k))[0]

            table.math_cell(0, k)
            table.array_cell(1, d_k)
            table.math_cell(2, '%.6f' % float(alpha_k))
            table.array_cell(3, x_kp1)
            table.array_cell(4, g_kp1)
            table.math_cell(5, '%.6f' % float(beta_k))

            # Update variables
            x_k = x_kp1
            g_k = g_kp1

            if g_kp1 == zero_vector:
                print('Stopping since g(k) = {}'.format(utils.format_vector(g_kp1)))
                break
        display(HTML(table.as_html()))
        return x_k
