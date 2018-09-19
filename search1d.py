import numpy as np
import matplotlib.pyplot as plt
import sympy as sy

from IPython.display import HTML, display

def plot_1d_function(x, y, xlimit=(-10, 10), ylimit=None, show_spines=True, figsize=(10, 6)):
    if ylimit is None:
        y_high = np.max(y)
        y_low = np.min(y)
        space = (y_high - y_low) / 20
        ylimit = (y_low - space, y_high + space)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, y)
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


class OneDimensionalIntervalBasedSearch:
    def __init__(self, f, x, initial_range, uncertainty, epsilon=0.0):
        self._f = sy.Matrix([f])
        self._x = x
        self._f_lambda = sy.lambdify(x, f)
        self._x_start = initial_range[0]
        self._x_end = initial_range[1]
        self._uncertainty = uncertainty
        self._epsilon = epsilon

    def plot(self, ylimit=None, show_spines=True, figsize=(10, 6)):
        x = np.linspace(self._x_start, self._x_end, 50)
        y = self._f_lambda(x)
        plot_1d_function(x, y, xlimit=(self._x_start, self._x_end), ylimit=ylimit, show_spines=show_spines, figsize=figsize)

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


class OneDimensionalIterativeSearch:
    def __init__(self, f, x):
        self._f = sy.Matrix([f])
        self._x = (x,)
        self._f_lambda = sy.lambdify(x, f)
        self._jacobian = self._f.jacobian(self._x)
        self._jacobian_lambda = sy.lambdify(self._x, self._jacobian)
        self._hessian = self._jacobian.jacobian(self._x)
        self._hessian_lambda = sy.lambdify(self._x, self._hessian)
        
    def func_at(self, point):
        """
        Evaluates the function f at the given point.
        """
        return self._f_lambda(point)
    
    def derivative_at(self, point):
        """
        Evaluates the first derivative f' at the given point.
        """
        return np.asscalar(self._jacobian_lambda(point))
    
    def second_derivative_at(self, point):
        """
        Evaluates the second derivative f'' at the given point.
        """
        return np.asscalar(self._hessian_lambda(point))

    def plot(self, xlimit=(-10, 10), ylimit=None, show_spines=True, figsize=(10,6)):
        x_start = xlimit[0]
        x_end = xlimit[1]
        x = np.linspace(x_start, x_end, 50)
        y = self.func_at(x)
        plot_1d_function(x, y, xlimit=xlimit, ylimit=ylimit, show_spines=show_spines, figsize=figsize)

    def __repr__(self):
        return self._f[0]

    def __str__(self):
        return str(self._f[0])

class NewtonSearch(OneDimensionalIterativeSearch):
    def __init__(self, f, x):
        super(NewtonSearch, self).__init__(f, x)

    def _update_for_minimisation(self, x_k):
        jacob_at_x_k = self.derivative_at(x_k)
        hess_at_x_k = self.second_derivative_at(x_k)
        return x_k - (jacob_at_x_k / hess_at_x_k)

    def _update_for_root_finding(self, x_k):
        f_at_x_k = self.func_at(x_k)
        jacob_at_x_k = self.derivative_at(x_k)
        return x_k - (f_at_x_k / jacob_at_x_k)
    
    def _run_algorithm(self, starting_point, epsilon, max_iterations, update_rule):
        x_k = starting_point
        has_converged = False
        
        # Run iterations
        for k in range(1, max_iterations):
            # Apply update rule
            x_kp1 = update_rule(x_k)
            
            # Check for convergence
            if np.abs(x_kp1 - x_k) < epsilon:
                has_converged = True
                
            # Update variables
            x_k = x_kp1
            
            print('Iteration {0:2}: x(k)={1}'.format(k, x_k))
            if has_converged:
                print(' Stopping condition reached!')
                break
        if not has_converged:
            print(' Stopping condition never reached!')
        
        return x_k

    def find_minimum(self, starting_point, epsilon=0.00001, max_iterations=21):
        return self._run_algorithm(
            starting_point, 
            epsilon, 
            max_iterations, 
            self._update_for_minimisation
        )
    
    def find_root(self, starting_point, epsilon=0.00001, max_iterations=21):
        return self._run_algorithm(
            starting_point, 
            epsilon, 
            max_iterations, 
            self._update_for_root_finding
        )

class SecantSearch(OneDimensionalIterativeSearch):
    def __init__(self, f, x):
        super(SecantSearch, self).__init__(f, x)

    def _update_for_minimisation(self, x_km1, x_k):
        jacob_at_x_k   = self.derivative_at(x_k)
        jacob_at_x_km1 = self.derivative_at(x_km1)
        return x_k - ((x_k - x_km1) / (jacob_at_x_k - jacob_at_x_km1)) * jacob_at_x_k

    def _update_for_root_finding(self, x_km1, x_k):
        f_at_x_k   = self.func_at(x_k)
        f_at_x_km1 = self.func_at(x_km1)
        return x_k - ((x_k - x_km1) / (f_at_x_k - f_at_x_km1)) * f_at_x_k
            
    def _run_algorithm(self, x_minus1, x_0, epsilon, max_iterations, update_rule):
        x_km1 = x_minus1
        x_k = x_0
        has_converged = False
        for k in range(1, max_iterations):
            # Apply update rule
            x_kp1 = update_rule(x_km1, x_k)
            
            # Check for convergence
            if np.abs(x_kp1 - x_k) < np.abs(x_k) * epsilon:
                has_converged = True
            
            # Update variables for next iteration
            x_km1 = x_k
            x_k   = x_kp1
            
            print('Iteration {0:2}: x({1})={2:.3f} x({3})={4:.3f}'.format(k, k-1, x_km1, k, x_k))
            if has_converged:
                print(' Stopping condition reached!')
                break
        if not has_converged:
            print(' Stopping condition never reached!')
        return x_k

    def find_minimum(self, x_minus1, x_0, epsilon=0.001, max_iterations=21):
        return self._run_algorithm(
            x_minus1,
            x_0,
            epsilon, 
            max_iterations, 
            self._update_for_minimisation
        )
    
    def find_root(self, x_minus1, x_0, epsilon=0.001, max_iterations=21):
        return self._run_algorithm(
            x_minus1,
            x_0,
            epsilon, 
            max_iterations, 
            self._update_for_root_finding
        )
