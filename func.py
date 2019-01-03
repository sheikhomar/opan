import numpy as np
import sympy as sy

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from mpl_toolkits.axisartist.axislines import SubplotZero
from IPython.display import display, HTML
from fractions import Fraction


class Func:
    def __init__(self, f, x, constraints=None):
        self._f = sy.Matrix([f])
        self._x = sy.Matrix(x)
        self._constraints = constraints
        self._f_lambda = sy.lambdify(self._x, self._f)
        self._jacobian = self._f.jacobian(self._x)
        self._jacobian_lambda = sy.lambdify(self._x, self._jacobian)
        self._hessian = self._jacobian.jacobian(self._x)
        self._hessian_lambda = sy.lambdify(self._x, self._hessian)

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
            return self._f.evalf(subs=params)

    def subs(self, point):
        params = dict(zip(self._x, point))
        return self._f.subs(params)

    def func_args(self):
        return self._x

    def func(self):
        return self._f

    def gradient(self):
        return self._jacobian.T

    def hessian(self):
        return self._hessian_lambda

    def func_at(self, point):
        params = dict(zip(self._x, point))
        return self.func().subs(params)

    def gradient_at(self, point):
        params = dict(zip(self._x, point))
        return self.gradient().subs(params)

    def hessian_at(self, point):
        params = dict(zip(self._x, point))
        return self._hessian.subs(params)

    def dimensions(self):
        return self._x.shape[0]

    def fonc_points(self):
        """
        Finds all the points that satisfy the First-Order Necessary Condition
        assuming that all the points are interior. This means that we need to
        solve the equation Df(x*)=0 for x*
        """
        return sy.nonlinsolve(self._jacobian, tuple(self._x))

    def satisfies_sonc(self, point):
        """
        Determines whether a point satisfies the Second-Order Necessary Condition.
        If the SONC is not satisfied then we have a saddle point.

        When the determinant of the Hessian matrix is positive, then we know that
        we have either a minimum or a maximum.

        The determinant of positive semidefinite is multiplication of its eigenvalues.
        """
        return self.hessian_at(point).det() >= 0

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
        all_params = tuple(self._x) + lambdas
        Dl = l.jacobian(all_params)

        # Solve it with respect to all the parameters
        result = sy.nonlinsolve(Dl, all_params)

        # Remove the lambda value from the results
        points = [tup[0:-num_constraints] for tup in list(result)]
        lambdas = [tup[-num_constraints:] for tup in list(result)]

        return points, lambdas

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

    def classify_critical_point(self, point):
        """
        Determines whether a given point is a saddle point.
        """
        # Compute the Hessian at the given point since it
        # tells us how the function behaves around the critical point.
        H_sympy = self.hessian_at(point)

        # Convert SymPy matrix to NumPy matrix
        H = np.array(H_sympy).astype(np.float64)

        # Compute the eigen values
        eigenvals = np.linalg.eigvals(H)

        # Check for positive and negative eigenvalues
        has_positive_eigenvals = len(eigenvals[eigenvals > 0]) > 0
        has_negative_eigenvals = len(eigenvals[eigenvals < 0]) > 0
        has_zero_eigenvals = len(eigenvals[eigenvals == 0]) > 0
        has_only_positive_eigenvals = has_positive_eigenvals and not has_negative_eigenvals
        has_only_negative_eigenvals = not has_positive_eigenvals and has_negative_eigenvals

        if has_zero_eigenvals:
            return "The point {} has a zero eigenvalue and so the test fails. The second order approximation isn't good enough and we need higher order information to make a decision.".format(point)
        
        # When the Hessian has both positive and negative
        # eigenvalues, the point is a saddle point
        # i.e. Hessian is indefinite
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

    def create(A, b, term3):
        x = sy.symbols('x1, x2', real=True)
        x_vec = np.array(x).reshape(2, -1)
        tmp = np.dot(x_vec.T, np.array(A))
        term1 = np.dot(tmp, x_vec)
        term2 = np.dot(x_vec.T, np.array(b).reshape(2, -1))
        result = term1 + term2 + term3
        return Func(result[0], x)

    def create_quadratic(Q, b, c=0):
        """
        Creates a quadratic function of the form:
         x.T*Q*x + x.T*b + c
        """
        return Func.create(Q, b, c)

    def create_quadratic_half(Q, b):
        """
        Creates a quadratic function of the form:
         1/2*x.T*Q*x  -  x.T*b
        """
        Q = sy.Matrix(Q)
        b = sy.Matrix(b)
        num_variables = int(Q.shape[0])
        x_symbols = sy.symbols('x1:'+ str(num_variables+1))
        x = sy.Matrix(x_symbols)
        f = sy.expand(Fraction(1, 2) * x.T * Q * x - x.T * b)[0]
        return Func(f, x_symbols)

    def taylor(self, x0):
        """
        Computes the third order Taylor series expansion about the
        given point x0.
        """
        # Compute the vector: x-x0 since we are going to use multiple times
        x_minus_x0  = self._x - sy.Matrix(x0)

        # Compute f(x0)
        term1 = self.func_at(x0)

        # Compute dfdx(x0)*(x-x0)
        term2 = self.gradient_at(x0).T * x_minus_x0

        # Compute p1 = 1/2*(x-x0)
        p1 = sy.Rational(1, 2) * x_minus_x0

        # Compute p2 = p1^T * Hessian(x0)
        p2 = p1.T * self.hessian_at(x0)

        # Compute term3 = p2 * (x-x0)
        term3 = p2 * x_minus_x0

        result = term1 + term2 + term3

        # Simplify the sum of all the terms
        # return sy.simplify(result)
        return result

    def parse_term(self, expr):
        syms = []
        cons = []

        if type(expr) == sy.Symbol:
            syms.append(expr)
            cons.append(1)
        elif type(expr) == sy.Pow:
            var, exponent = expr.args
            if type(var) == sy.Symbol:
                syms.append(expr)
                cons.append(1)
            else:
                cons.append(expr)
        else:
            for arg in expr.args:
                if type(arg) == sy.Symbol:
                    syms.append(arg)
                elif type(arg) == sy.Pow:
                    var, exponent = arg.args
                    if type(var) == sy.Symbol:
                        syms.append(arg)
                    else:
                        cons.append(arg)
                else:
                    cons.append(arg)

        const_term = 1
        syms_term = None
        if len(cons) == 1:
            const_term = cons[0]
        elif len(cons) > 1:
            const_term = sy.Mul(*cons)
        if len(syms) == 1:
            syms_term = syms[0]
        elif len(syms) > 0:
            syms_term = sy.Mul(*syms)
        return const_term, syms_term

    def as_quadratic_form_for_gradient_descent(self):
        """
        Rewrites the given function f as quadratic form that is
        use for gradient descent algorithms.

        The matrix Q and vector b is equal to f (except the constant term) if
        following expression is computed:

          1/2 * np.dot(np.dot(x.T, Q), x) - np.dot(b.T, x)
        """
        f = sy.expand(self._f[0])
        x1 = self._x[0]
        x2 = self._x[1]
        a, b_or_c, d, b1, b2 = 0, 0, 0, 0, 0

        for term in f.args:
            const_term, syms_term = self.parse_term(term)

            if syms_term is None:
                print('Ignoring constant term: {}'.format(const_term))
            elif type(syms_term) == sy.Pow:
                var, exponent = syms_term.args
                if not exponent == 2:
                    raise Exception('Expected exponent to be 2 but was ' + str(exponent))
                if var == x1:
                    a = const_term
                elif var == x2:
                    d = const_term
                else:
                    raise Exception('Invalid variable found: ' + str(var))
            elif type(syms_term) == sy.Mul:
                var1, var2 = syms_term.as_two_terms()
                if (var1 == x1 and var2 == x2) or (var2 == x1 and var1 == x2):
                    b_or_c = const_term
                else:
                    raise Exception('Cannot handle product: ' + str(syms_term))
            elif type(syms_term) == sy.Symbol:
                if syms_term == x1:
                    b1 = -1 * const_term
                elif syms_term == x2:
                    b2 = -1 * const_term
                else:
                    raise Exception('Contains an invalid product: ' + str(syms_term))
            else:
                raise Exception('Contains an invalid product: ' + str(syms_term))
        # Build matrix Q and vector b so f is (except the constant term)
        # f ~= 1/2 * np.dot(np.dot(x.T, Q), x) - np.dot(b.T, x)
        Q = sy.Matrix([[a*2,     b_or_c],
                      [b_or_c,  d*2]])
        b = sy.Matrix([[b1],
                       [b2]])
        return Q, b

    def find_step_size_range(self, Q=None):
        """
        Find the largest range of values of alpha for which
        the fixed-step-size gradient descent algorithm is
        globally convergent.
        """
        if Q is None:
            Q, b = self.as_quadratic_form_for_gradient_descent()

        eigenvalues = list(Q.eigenvals().keys())
        lambda_max = np.max(eigenvalues)
        upperbound = 2/lambda_max
        return HTML('$0 < \\alpha < {}$'.format(upperbound))

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

    def plot_contour(self, figsize=(12,6), x_limit=(-4, 4), delta=0.0025, levels=10, filled=True):
        if len(self._x) != 2:
            raise Exception('Cannot draw contour plot with {0} variables.'.format(len(self._x)))

        x1_vals = np.arange(x_limit[0], x_limit[1], delta)
        x2_vals = np.arange(x_limit[0], x_limit[1], delta)
        X1, X2 = np.meshgrid(x1_vals, x2_vals)

        # Compute Z values
        Z = self._f_lambda(X1, X2)

        # Reshape Z so it is 2D matrix
        Z = Z.reshape(Z.shape[2], Z.shape[2])

        fig, ax = plt.subplots(figsize=figsize)
        conourSet1 = ax.contourf(X1, X2, Z, levels) if filled else ax.contour(X1, X2, Z, levels)
        fig.colorbar(conourSet1, ax=ax)
        ax.set_title('Contour plot for ${}$'.format(sy.latex(self._f[0])))
        return fig, ax

    def __repr__(self):
        display(self._f)
        return ''

    def __str__(self):
        return str(self._f)


