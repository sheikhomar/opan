import numpy as np
import matplotlib.pyplot as plt
import sympy as sy

from mpl_toolkits.axisartist.axislines import SubplotZero
from IPython.display import display, HTML


import matplotlib.animation as animation

def create_animation(fig, ax, points):
    line1, = ax.plot([], [], 'go', lw=2)
    line2, = ax.plot([], [], 'rx', lw=2)
    
    def _init():
        line1.set_data([], [])
        line2.set_data([], [])
        return line1, line2
    
    def _animate(k):
        x_k, z_k = points[k]
        line1.set_data(x_k[0], x_k[1])
        line2.set_data(z_k[0], z_k[1])
        return line1, line2
    
    # blit=True redraws regions that have changed
    anim = animation.FuncAnimation(fig, _animate, init_func=_init,
                               frames=len(points), interval=500, blit=True)
    return anim



class NaiveRandomSearch:
    def __init__(self, f):
        self._f = f
    
    def check_feasible_set(self):
        feasible_set = [
            sy.Interval(-3, sy.oo),
            sy.Interval(-sy.oo, 3)
        ]

        def fetch_val(val, min_val=-1000, max_val=1000):
            if val == sy.oo:
                return max_val
            if val == -sy.oo:
                return min_val
            return val

        low = np.zeros(len(feasible_set))
        high = np.zeros(len(feasible_set))
        for index, interval in enumerate(feasible_set):
            if isinstance(interval, sy.EmptySet):
                raise ValueError('Interval at {0} is invalid.'.format(index))
            low[index] = fetch_val(interval.left)
            high[index] = fetch_val(interval.right)

        low, high  
    
    def pick_random_point(self, x_k, alpha):
        # TODO: Ensure that these points are within the feasible set
        low  = x_k - alpha
        high = x_k + alpha
        return np.random.uniform(low, high)
    
    def f_at(self, point):
        f_val = self._f.func_at(point)
        return f_val[0]
    
    def run(self, initial_point, alpha, max_iterations=20, max_tries=10, verbose=True):
        iterates = []
        has_converged = False
        ntries = 0
        
        x_k = np.array(initial_point)
        for k in range(1, max_iterations+1):
            z_k = self.pick_random_point(x_k, alpha)
            
            f_at_z_k = self.f_at(z_k)
            f_at_x_k = self.f_at(x_k)
            
            if self.f_at(z_k) < self.f_at(x_k):
                x_kp1 = z_k
            else:
                x_kp1 = x_k
            
            ntries = ntries + 1 if np.array_equal(x_kp1, x_k) else 0
            if ntries > max_tries:
                has_converged = True
            
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
                if has_converged:
                    print('Point has not improved for {} iterations. Stopped!'.format(max_tries))
            
            iterates.append((x_k, z_k))
            x_k = x_kp1
            
            if has_converged:
                break
            
        return x_k, iterates
