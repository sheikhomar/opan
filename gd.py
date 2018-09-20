import numpy as np
import sympy as sy

def rewrite_as_quadratic_form(f, x1, x2):
    """
    Rewrites the given function f as quadratic form and returns Q and b.

    The matrix Q and vector b is equal to f (except the constant term) if
    following expression is computed:

      1/2 * np.dot(np.dot(x.T, Q), x) - np.dot(b.T, x)
    """
    a, b_or_c, d, b1, b2 = 0, 0, 0, 0, 0

    for term in f.args:
        if type(term) == sy.Mul:
            coeff, prod = term.as_two_terms()
            if type(prod) == sy.Pow:
                var, exponent = prod.args
                if not exponent == 2:
                    raise Exception('Expected exponent to be 2 but was ' + str(exponent))
                if var == x1:
                    a = coeff
                elif var == x2:
                    d = coeff
                else:
                    raise Exception('Invalid variable found: ' + str(var))
            elif type(prod) == sy.Mul:
                var1, var2 = prod.as_two_terms()
                if (var1 == x1 and var2 == x2) or (var2 == x1 and var1 == x2):
                    b_or_c = coeff
                else:
                    raise Exception('Cannot handle product: ' + str(prod))
            elif type(prod) == sy.Symbol:
                if prod == x1:
                    b1 = -1 * coeff
                elif prod == x2:
                    b2 = -1 * coeff
                else:
                    raise Exception('Contains an invalid product: ' + str(prod))
            else:
                raise Exception('Contains an invalid product: ' + str(prod))
        elif type(term) == sy.Symbol:
            if term == x1:
                b1 = -1
            elif term == x2:
                b2 = -1
            else:
                raise Exception('Contains an invalid term: ' + str(term))
        else:
            print('Ignoring constant term: ' + str(term))

    # Build matrix Q and vector b so f is (except the constant term)
    # f ~= 1/2 * np.dot(np.dot(x.T, Q), x) - np.dot(b.T, x)
    Q = sy.Matrix([[a*2,     b_or_c],
                  [b_or_c,  d*2]])
    b = sy.Matrix([[b1],
                   [b2]])
    return Q, b
