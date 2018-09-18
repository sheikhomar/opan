import numpy as np
import matplotlib.pyplot as plt
import sympy as sy

from IPython.display import HTML, display

class OneDimensionalIntervalBasedSearch:
    def __init__(self, f, x, initial_range, uncertainty, epsilon=0.0):
        self._f = sy.Matrix([f])
        self._x = x
        self._f_lambda = sy.lambdify(x, f)
        self._x_start = initial_range[0]
        self._x_end = initial_range[1]
        self._uncertainty = uncertainty
        self._epsilon = epsilon

    def plot(self, ylimit=None):
        x = np.linspace(self._x_start, self._x_end, 50)
        y = self._f_lambda(x)

        if ylimit is None:
            y_high = np.max(y)
            y_low = np.min(y)
            space = (y_high - y_low) / 20
            ylimit = (y_low - space, y_high + space)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, y)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$f(x)$')
        ax.set_xlim(self._x_start, self._x_end)
        ax.set_ylim(*ylimit)

    def run(self, num_iterations):
        html_out = (
            '<table><thead><tr>'
            '<th>$k$</th>'
            '<th>$a_k$</th>'
            '<th>$b_k$</th>'
            '<th>$f(a_k)$</th>'
            '<th>$f(b_k)$</th>'
            '<th>Uncertainty interval</th>'
            '</tr></thead>'
            '<tbody>'
        )

        # Set our initial interval
        x_start = self._x_start
        x_end   = self._x_end

        for k in range(1, num_iterations+1):
            # Compute rho
            rho = self._compute_rho(k, num_iterations)

            # Find two intermediate points a and b within the current
            #  uncertainty interval using the Golden section rho
            a = x_start +     rho * (x_end - x_start)
            b = x_start + (1-rho) * (x_end - x_start)

            # When using the Fibonacci search, at the final stage,
            # the two intermediate points a and b coincide in the
            # middle of the uncertainty interval. To get around this
            # problem, we need to nudge the point a with a small
            # number, episolon.
            if a == b:
                a = x_start + (rho - self._epsilon) * (x_end - x_start)

            # Evaluate f at the intermediate points
            fa = self._f_lambda(a)
            fb = self._f_lambda(b)

            uncert_int = (0, 0)

            # Determine the new uncertainty interval
            if fa < fb:
                # If f(a) < f(b), then we should move our
                # current uncertainty interval from the right
                uncert_int = (x_start, b)
                x_end = b # Move the right interval
            elif fa > fb:
                # If f(a) > f(b), then we should move our
                # current uncertainty interval from the left
                uncert_int = (a, x_end)
                x_start = a # Move the left interval

            html_out += (
                '<tr>'
                '<td>{0}</td>'
                '<td>{1:5.4f}</td>'
                '<td>{2:5.4f}</td>'
                '<td>{3:5.4f}</td>'
                '<td>{4:5.4f}</td>'
                '<td>[{5:5.4f}, {6:5.4f}]</td>'
                '</tr>'
            ).format(k, a, b, fa, fb, uncert_int[0], uncert_int[1])

        html_out += '</tbody></table>'
        display(HTML(html_out))

    def _compute_rho(self, k, num_iterations):
        raise Exception('Not implemented')


class GoldenSectionSearch(OneDimensionalIntervalBasedSearch):
    def __init__(self, f, x, initial_range, uncertainty):
        super(GoldenSectionSearch, self).__init__(f, x, initial_range, uncertainty)

        # Compute rho once
        self._rho = (3-np.sqrt(5))/2

    def estimate_iterations(self):
        """
        Estimates the number of iterations required to find the
        value of x within a range of the specified uncertainty.
        """
        N = sy.symbols('N')
        range_diff = self._x_end - self._x_start
        ineq = (0.61803)**N <= self._uncertainty / range_diff
        interval = sy.solveset(ineq, N, sy.S.Reals)
        interval_start = int(np.ceil(float(interval.args[0])))
        return interval_start

    def _compute_rho(self, k, num_iterations):
        return self._rho


class FibonacciSearch(OneDimensionalIntervalBasedSearch):

    def __init__(self, f, x, initial_range, uncertainty, epsilon):
        super(FibonacciSearch, self).__init__(
            f, x, initial_range, uncertainty, epsilon)

    def estimate_iterations(self):
        """
        Estimates the number of iterations required to find the
        value of x within a range of the specified uncertainty.
        """
        N = sy.symbols('N')
        initial_range = self._x_end - self._x_start
        final_range = self._uncertainty

        ineq = N >= (1 + 2*self._epsilon) / (final_range / initial_range)
        interval = sy.solveset(ineq, N, sy.S.Reals)
        interval_start = int(np.ceil(float(interval.args[0])))

        for i in range(2, 10):
            if interval_start <= sy.fibonacci(i):
                return i-2

        raise ValueError('Could not find the correct Fibonacci number.')

    def _compute_rho(self, k, num_iterations):
        # The l variable counts backwards whereas k counts forwards.
        l = num_iterations - (k-1)

        # Compute rho using Fibonacci numbers
        result = 1 - sy.fibonacci(l+1) / sy.fibonacci(l+2)

        return float(result)


class NewtonSearch:
    def __init__(self, f, x):
        self._f = sy.Matrix([f])
        self._x = (x,)
        self._f_lambda = sy.lambdify(x, f)
        self._jacobian = self._f.jacobian(self._x)
        self._jacobian_lambda = sy.lambdify(self._x, self._jacobian)
        self._hessian = self._jacobian.jacobian(self._x)
        self._hessian_lambda = sy.lambdify(self._x, self._hessian)

    def evaluate(self, x):
        return self._f_lambda(x)

    def run(self, starting_point, epsilon, max_iterations=21):
        x_k = starting_point
        stop = False
        for k in range(1, max_iterations):
            jacob_at_x_k = np.asscalar(self._jacobian_lambda(x_k))
            hess_at_x_k = np.asscalar(self._hessian_lambda(x_k))
            x_kp1 = x_k - (jacob_at_x_k / hess_at_x_k)
            if np.abs(x_kp1 - x_k) < epsilon:
                stop = True
            x_k = x_kp1
            print('Iteration {0:2}: x(k)={1}'.format(k, x_k))
            if stop:
                print(' Stopping condition reached!')
                break
        if not stop:
            print(' Stopping condition never reached!')
        return x_k

    def find_root(self, starting_point, epsilon=0.00001, max_iterations=21):
        x_k = starting_point
        stop = False
        for k in range(1, max_iterations):
            f_at_x_k = self._f_lambda(x_k)
            jacob_at_x_k = np.asscalar(self._jacobian_lambda(x_k))
            x_kp1 = x_k - (f_at_x_k / jacob_at_x_k)
            if np.abs(x_kp1 - x_k) < epsilon:
                stop = True
            x_k = x_kp1
            print('Iteration {0:2}: x(k)={1}'.format(k, x_k))
            if stop:
                print(' Stopping condition reached!')
                break
        if not stop:
            print(' Stopping condition never reached!')
        return x_k

    def plot(self, xlimit=(-10, 10), ylimit=None, show_spines=False):
        x_start = xlimit[0]
        x_end = xlimit[1]
        x = np.linspace(x_start, x_end, 50)
        y = self._f_lambda(x)

        if ylimit is None:
            y_high = np.max(y)
            y_low = np.min(y)
            space = (y_high - y_low) / 20
            ylimit = (y_low - space, y_high + space)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, y)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$f(x)$')
        ax.set_xlim(x_start, x_end)
        ax.set_ylim(*ylimit)

        if show_spines:
            ax.spines['left'].set_position('zero')
            ax.spines['right'].set_color('none')
            ax.spines['bottom'].set_position('zero')
            ax.spines['top'].set_color('none')
            ax.spines['left'].set_smart_bounds(True)
            ax.spines['bottom'].set_smart_bounds(True)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')

