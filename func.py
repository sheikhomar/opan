import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axisartist.axislines import SubplotZero
import sympy as sy
from IPython.display import display


class Func:
    def __init__(self, f, x, constraints=None):
        self._f = sy.Matrix([f])
        self._x = x
        self._constraints = constraints
        self._f_lambda = sy.lambdify(x, self._f)
        self._jacobian = self._f.jacobian(x)
        self._jacobian_lambda = sy.lambdify(x, self._jacobian)
        self._hessian = self._jacobian.jacobian(x)
        self._hessian_lambda = sy.lambdify(x, self._hessian)

    def evalf(self, points):
        """
        Evaluates the function at a certain point.
        """
        if isinstance(points, list):
            for point in points:
                params = dict(zip(self._x, point))
                result = self._f.evalf(subs=params)
                print('Evaluting f{} = {}'.format(point, result[0]))
        else:
            params = dict(zip(self._x, points))
            return self._f.evalf(subs=params),

    def solve_lagrangian(self):
        num_constraints = len(self._constraints)

        # Define a lambda symbol for each constraint
        lambda_names = tuple([('lambda%s' % i) for i in range(1, num_constraints+1)])
        lambdas = sy.symbols(lambda_names)
        lambda_vec = sy.Matrix(lambdas)

        # Define constraints as vector
        H = sy.Matrix(self._constraints)

        # Formulate the Lagrangian function l
        l = self._f + lambda_vec.T * H

        # Find the gradient of the Lagrangian function
        # with respect to all parameters
        all_params = self._x + lambdas
        Dl = l.jacobian(all_params)

        # Solve it with respect to all the parameters
        result = sy.nonlinsolve(Dl, all_params)

        # Remove the lambda value from the results
        points = [tup[0:-num_constraints] for tup in list(result)]
        lambdas = [tup[-num_constraints:] for tup in list(result)]

        return points, lambdas

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
        If the SONC is not satisfied then we have a saddle point.

        When the determinant of the Hessian matrix is positive, then we know that
        we have either a minimum or a maximum.

        The determinant of positive semidefinite is multiplication of its eigenvalues.
        """
        return self.hessian_at(point).det() >= 0

    def plot_point(self, point):
        x_limit = 10
        y_limit = 5
        fig = plt.figure(1, figsize=(8,4))
        ax = SubplotZero(fig, 1, 1, 1)
        fig.add_subplot(ax)
        ax.grid(color='#eeeeee', linestyle='-', linewidth=2)
        ax.set_xticks(np.arange(-1*x_limit, x_limit, 1))
        ax.set_xlim(-1*x_limit, x_limit)
        ax.set_yticks(np.arange(-1*y_limit, y_limit, 1))
        ax.set_ylim(-1*y_limit, y_limit)
        ax.axis['xzero'].set_axisline_style('-|>')
        ax.axis['xzero'].set_visible(True)
        ax.axis["xzero"].label.set_text('$x_1$')
        ax.axis['yzero'].set_axisline_style('-|>')
        ax.axis['yzero'].set_visible(True)
        ax.axis["yzero"].label.set_text('$x_2$')
        ax.plot(*point,'ro',label='$x^{*}$')
        ax.legend();

    def fonc_at(self, point):
        """
        Computes the First-Order Necessary Condition at a given point.
        """
        grad = self.gradient_at(point)
        d1, d2 = sy.symbols('d1, d2')
        fonc_expr = d1 * grad[0]  + d2 * grad[1] >= 0
        return fonc_expr

    def find_critical_points(self):
        """
        Finds all the points where the gradient is zero and classifies them.
        """
        points = self.fonc_points()
        for point in points:
            print(self.classify_critical_point(point))
        return points

    def classify_critical_point2(self, point):
        """
        Determines whether a given point is a saddle point.
        This is the same as checking whether a point satisfies the Second-Order
        Necessary Condition.
        """
        # Compute the Hessian at the given point
        H_sympy = self.hessian_at(point)

        # Convert SymPy matrix to NumPy matrix
        H = np.array(H_sympy).astype(np.float64)

        # Compute the eigen values
        eigenvals = np.linalg.eigvals(H)

        # Check of positive and negative eigenvalues
        has_positive_eigenvals = len(eigenvals[eigenvals >= 0]) > 0
        has_negative_eigenvals = len(eigenvals[eigenvals < 0]) > 0
        has_only_positive_eigenvals = has_positive_eigenvals and not has_negative_eigenvals
        has_only_negative_eigenvals = not has_positive_eigenvals and has_negative_eigenvals

        # When the Hessian has both positive and negative
        # eigenvalues, the point is a saddle point
        # i.e. Hessian is not positive semidefinite
        if has_positive_eigenvals and has_negative_eigenvals:
            return 'The point {} is a saddle point.'.format(point)

        # When the Hessian has only positive eigenvalues
        # then the point is a local minimum.
        # Hessian is positive semidefinite
        if has_only_positive_eigenvals:
            return 'The point {} is a local minimum.'.format(point)

        # When the Hessian has only negative eigenvalues
        # then the point is a local maximum.
        # Hessian is negative definite
        if has_only_negative_eigenvals:
            return 'The point {} is a local maximum.'.format(point)

        return 'Unknown'

    def classify_critical_point(self, point):
        """
        Determines whether a given point is a saddle point.
        This is the same as checking whether a point satisfies the Second-Order
        Necessary Condition.
        """
        # Compute the Hessian at the given point
        H = self.hessian_at(point)

        # If the determinant of the Hessian matrix is negative,
        # then we have a saddle point.
        if H.det() < 0:
            return 'The point {} is a saddle point.'.format(point)

        # If the determinant of the Hessian matrix is positive,
        # then we know that we have either a minimum or a maximum.

        # If the first entry of the Hessian is positive, then we have a minimum.
        if H[0,0] > 0:
            return 'The point {} is a local minimum.'.format(point)

        # If the first entry of the Hessian is negative or zero, then we have a maximum.
        return 'The point {} is a local maximum.'.format(point)

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
        plot.xlabel = self._x[0]
        plot.ylabel = self._x[1]
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

    def __repr__(self):
        display(self._f)
        return ''

    def __str__(self):
        return str(self._f)
