import numpy as np
import matplotlib.pyplot as plt
import sympy as sy

class Func:
    def __init__(self, f, x):
        self._f = sy.Matrix([f])
        self._x = x
        self._f_lambda = sy.lambdify(x, self._f)
        self._jacobian = self._f.jacobian(x)
        self._jacobian_lambda = sy.lambdify(x, self._jacobian)
        self._hessian = self._jacobian.jacobian(x)
        self._hessian_lambda = sy.lambdify(x, self._hessian)

    def gradient(self):
        return self._jacobian

    def hessian(self):
        return self._hessian

    def gradient_at(self, point):
        return self._jacobian.subs({
            self._x[0]: point[0],
            self._x[1]: point[1]
        })

    def hessian_at(self, point):
        return self._hessian.subs({
            self._x[0]: point[0],
            self._x[1]: point[1]
        })

    def rate_of_increase(self, point):
        gradient_at_point = sy.lambdify(self._x, self._jacobian)(*tuple(point))
        unit_vector = gradient_at_point / np.linalg.norm(gradient_at_point)
        return np.dot(gradient_at_point, unit_vector.T)

    def fonc_points(self):
        """
        Finds all the points that satisfy the First-Order Necessary Condition
        assuming that all the points are interior. This means that we need to
        solve the equation Df(x*)=0 for x*
        """
        return sy.nonlinsolve(self._jacobian, self._x)

    def satisfies_sonc(self, point):
        """
        Determines whether a point satisfies the Second-Order Necessary Condition.
        """
        return self.hessian_at(point).det() >= 0

    def plot(self, range_x=None, range_y=None):
        if range_x is not None and len(range_x) == 2:
            range_x = (self._x[0], range_x[0], range_x[1])
        else:
            range_x = (self._x[0], -10, 10)
        if range_y is not None and len(range_y) == 2:
            range_y = (self._x[1], range_y[0], range_y[1])
        else:
            range_y = (self._x[1], -10, 10)
        plot = sy.plotting.plot3d(self._f[0], range_x, range_y, show=False, legend=False)
        plot.xlabel = '$x_1$'
        plot.ylabel = '$x_2$'
        plot.show()

    def create(A, b, term3):
        x = sy.symbols('x1, x2', real=True)
        x_vec = np.array(x).reshape(2, -1)
        tmp = np.dot(x_vec.T, np.array(A))
        term1 = np.dot(tmp, x_vec)
        term2 = np.dot(x_vec.T, np.array(b).reshape(2, -1))
        result = term1 + term2 + term3
        return Func(result[0], x)

    def taylor(self, x0):
        """
        Computes the third order Taylor series expansion about the
        given point x0.
        """
        # Compute the vector: x-x0 since we are going to use multiple times
        x_minus_x0 = (np.array(self._x) - np.array(x0)).reshape(2, -1)

        # Compute f(x0)
        term1 = self._f_lambda(*x0)

        # Compute dfdx(x0)*(x-x0)
        term2 = np.dot(self._jacobian_lambda(*x0), x_minus_x0)

        # Compute p1 = 1/2*(x-x0)
        p1 = sy.Rational(1, 2) * x_minus_x0

        # Compute p2 = p1^T * Hessian(x0)
        p2 = np.dot(p1.T, self._hessian_lambda(*x0))

        # Compute term3 = p2 * (x-x0)
        term3 = np.dot(p2, x_minus_x0)

        result = (term1 + term2 + term3)[0][0]

        # Simplify the sum of all the terms
        simplified_result = sy.simplify(term1 + term2 + term3)[0]

        return simplified_result