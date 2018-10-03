import numpy as np
import matplotlib.pyplot as plt
import sympy as sy

from mpl_toolkits.axisartist.axislines import SubplotZero
from IPython.display import display, HTML


import matplotlib.animation as animation

def create_animation(fig, ax, result):
    points = list(zip(result['x_vals'], result['z_vals']))

    line1, = ax.plot([], [], 'go', lw=2)
    line2, = ax.plot([], [], 'rx', lw=2)

    def _init():
        line1.set_data([], [])
        line2.set_data([], [])
        return line1, line2

    def _animate(k):
        x_k = result['x_vals'][k]
        z_k = result['z_vals'][k]
        line1.set_data(x_k[0], x_k[1])
        line2.set_data(z_k[0], z_k[1])
        return line1, line2

    # blit=True redraws regions that have changed
    anim = animation.FuncAnimation(fig, _animate, init_func=_init,
                               frames=len(points), interval=500, blit=True)
    return anim



class RandomizedSearch:
    def __init__(self, f, feasible_set):
        self._f = f
        self._feasible_set = np.array(feasible_set)

    def pick_candidate(self, x_k, alpha):
        """
        Pick a candidate point at random from the neighbourhood of x(k)
        """
        # Compute the range of the random value for each entry
        low  = x_k - alpha
        high = x_k + alpha

        # Ensure the min values are within the feasible set
        feasible_low = self._feasible_set[:,0]
        a = np.concatenate([feasible_low, low]).reshape(2, 2)
        low = np.amax(a, axis=0)

        # Ensure the max values are within the feasible set
        feasible_high = self._feasible_set[:,1]
        a = np.concatenate([feasible_high, high]).reshape(2, 2)
        high = np.amin(a, axis=0)

        return np.random.uniform(low, high)

    def f_at(self, point):
        f_val = self._f.func_at(point)
        return f_val[0]



class NaiveRandomSearch(RandomizedSearch):
    def __init__(self, f, feasible_set):
        super(NaiveRandomSearch, self).__init__(f, feasible_set)

    def run(self, initial_point, alpha, max_iterations=20, max_tries=10, verbose=True):
        ret_val = {
            'x_vals': [],
            'z_vals': [],
            'best_point': None,
            'best_cost': None,
            'has_converged': False,
        }
        ntries = 0

        x_k = np.array(initial_point)
        for k in range(1, max_iterations+1):
            z_k = self.pick_candidate(x_k, alpha)

            f_at_z_k = self.f_at(z_k)
            f_at_x_k = self.f_at(x_k)

            if f_at_z_k < f_at_x_k:
                x_kp1 = z_k
            else:
                x_kp1 = x_k

            # Track x(k) and z(k)
            ret_val['x_vals'].append(x_k)
            ret_val['z_vals'].append(z_k)

            # Update best best point and cost so far
            f_at_x_kp1 = self.f_at(x_kp1)
            if ret_val['best_cost'] is None or ret_val['best_cost'] > f_at_x_kp1:
                ret_val['best_cost'] = f_at_x_kp1
                ret_val['best_point'] = x_kp1

            ntries = ntries + 1 if np.array_equal(x_kp1, x_k) else 0
            if ntries > max_tries:
                ret_val['has_converged'] = True

            if verbose:
                print('k={0:2}: x[k]={1} f(x[k])={2:.4f}  z(k)={3}, f(z[k])={4:.4f} x[k+1]={5}'.format(
                    k,
                    utils.format_vector(x_k),
                    float(f_at_x_k),
                    utils.format_vector(z_k),
                    float(f_at_z_k),
                    utils.format_vector(x_kp1)
                    )
                )
                if ret_val['has_converged']:
                    print('Point has not improved for {} iterations. Stopped!'.format(max_tries))

            x_k = x_kp1

            if ret_val['has_converged']:
                break

        return ret_val


class SimulatedAnnealing(RandomizedSearch):
    def __init__(self, f, feasible_set):
        super(SimulatedAnnealing, self).__init__(f, feasible_set)

    def get_temp(self, k, gamma=1):
        """
        Computes temperature at iteration k based on the
        desired cooling schedule.
        """
        return gamma / np.log(k + 2)

    def run(self, initial_point, alpha, gamma=2, max_iterations=20, max_tries=10, verbose=True):
        x_k = np.array(initial_point)
        ntries = 0
        ret_val = {
            'x_vals': [],
            'z_vals': [],
            'temps': [],
            'best_point': None,
            'best_cost': None,
            'has_converged': False,
        }

        for k in range(1, max_iterations+1):
            # Pick a candidate point
            z_k = self.pick_candidate(x_k, alpha)

            # Compute the cost of the candidate point
            f_at_z_k = self.f_at(z_k)

            # Compute the cost of current point
            f_at_x_k = self.f_at(x_k)

            # Get the temperature at iteration k
            temp = self.get_temp(k)

            # Compute the acceptance probability
            p = np.exp(- float(f_at_z_k - f_at_x_k) / temp)

            # Random value in the interval [0.0, 1.0)
            r = np.random.random_sample()

            if p > r:
                x_kp1 = z_k
            else:
                x_kp1 = x_k

            ret_val['x_vals'].append(x_k)
            ret_val['z_vals'].append(z_k)
            ret_val['temps'].append(temp)

            # Update best best point and cost so far
            f_at_x_kp1 = self.f_at(x_kp1)
            if ret_val['best_cost'] is None or ret_val['best_cost'] > f_at_x_kp1:
                ret_val['best_cost'] = f_at_x_kp1
                ret_val['best_point'] = x_kp1

            # Check for convergence
            ntries = ntries + 1 if np.array_equal(x_kp1, x_k) else 0
            if ntries > max_tries:
                ret_val['has_converged'] = True

            if verbose:
                print('k={0:2}: x[k]={1} f(x[k])={2:.4f}  z(k)={3}, f(z[k])={4:.4f} x[k+1]={5}'.format(
                    k,
                    utils.format_vector(x_k),
                    float(f_at_x_k),
                    utils.format_vector(z_k),
                    float(f_at_z_k),
                    utils.format_vector(x_kp1)
                    )
                )
                if ret_val['has_converged']:
                    print('Point has not improved for {} iterations. Stopped!'.format(max_tries))

            x_k = x_kp1


            if ret_val['has_converged']:
                break

        return ret_val

