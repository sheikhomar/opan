from scipy.optimize import linprog

import numpy as np
import fractions
np.set_printoptions(formatter={'all':lambda x: str(fractions.Fraction(x).limit_denominator())})


def maximise(A, b, c):
    # Convert the objective function because the formulation 
    # of the linear programming problem in SciPy deviates from the 
    # standard linear programming problem.
    c = np.array(c) * -1 

    # Create x bounds
    A = np.array(A)
    num_variables = A.shape[1]
    bounds = tuple((0, None) for i in range(num_variables)) # x_i >= 0

    # Use Scipy to compute the result
    res = linprog(c, A, b, bounds=bounds)

    # Turn the value of the objective function 
    # back to maximum
    res.fun = res.fun * -1

    return res


def minimise(A, b, c):
    # Change inequalities
    A = np.array(A) * -1
    b = np.array(b) * -1

    # Create x bounds
    num_variables = A.shape[1]
    bounds = tuple((0, None) for i in range(num_variables)) # x_i >= 0

    # Use Scipy to compute the result
    res = linprog(c, A, b, bounds=bounds)

    return res
