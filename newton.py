import numpy as np
import matplotlib.pyplot as plt
import sympy as sy
import utils as utils

from func import Func

from IPython.display import HTML, display

class NewtonMultivariateSearch:
    def __init__(self, f: Func):
        self._f = f

    def _update_for_minimisation_using_solve(self, x_k):
        func = self._f
        x = func.func_args()
        
        jacob_at_x_k = func.gradient_at(x_k)
        hess_at_x_k = func.hessian_at(x_k)
        delta_x = x - x_k
        
        # Setup system of equation
        system_equation = hess_at_x_k * delta_x + jacob_at_x_k
        
        # Solve it the system of equations
        result = sy.nonlinsolve(system_equation, list(x))
        
        # Convert from set to vector
        res_vector = sy.Matrix(list(result)[0])
        
        return x_k - res_vector
        
    def _update_for_minimisation(self, x_k: sy.Matrix):
        jacob_at_x_k = self._f.gradient_at(x_k)
        hess_at_x_k = self._f.hessian_at(x_k)
        hess_inv = hess_at_x_k.inv()
        return x_k - hess_inv * jacob_at_x_k

    def _update_for_root_finding(self, x_k: sy.Matrix):
        f_at_x_k = self._f.func_at(x_k)
        jacob_at_x_k = self._f.gradient_at(x_k)
        jacob_inv = jacob_at_x_k.inv()
        return x_k  - jacob_inv * f_at_x_k
    
    def _stopping_criteria_for_minimisation(self, x_k, x_kp1, epsilon: float):
        # For minimisation, we can stop if the increment size 
        # from x(k) to x(k+1) is small enough so we will stop.
        
        # Compute increment size: ||x(k+1) - x(k)||
        increment_size = (x_kp1 - x_k).norm()
        
        return increment_size < epsilon

    def _stopping_criteria_for_root_finding(self, x_k, x_kp1, epsilon: float):
        # For root finding, we can stop when f(x_k) is close
        # to zero e.g. |f(x_k)| < epsilon
        return self._f.func_at(x_k) < epsilon
    
    def _run_algorithm(self, starting_point, epsilon: float, max_iterations: int, update_rule, convergence_rule, verbose=True):
        x_k = sy.Matrix(starting_point)
        has_converged = False

        # Run iterations
        for k in range(1, max_iterations+1):
            
            # Apply update rule
            x_kp1 = update_rule(x_k)
            
            # Check for convergence
            has_converged = convergence_rule(x_k, x_kp1, epsilon)
            
            # Update variables
            x_k = x_kp1
            
            print('Iteration {0:2}: x({0})={1} '.format(
                k, utils.format_vector(x_k) )
            )
            
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
            self._update_for_minimisation,
            self._stopping_criteria_for_minimisation
        )

    def find_minimum_using_solve(self, starting_point, epsilon=0.00001, max_iterations=21):
        return self._run_algorithm(
            starting_point,
            epsilon,
            max_iterations,
            self._update_for_minimisation_using_solve,
            self._stopping_criteria_for_minimisation
        )
    
    def find_root(self, starting_point, epsilon=0.00001, max_iterations=21):
        return self._run_algorithm(
            starting_point,
            epsilon,
            max_iterations,
            self._update_for_root_finding,
            self._stopping_criteria_for_root_finding
        )