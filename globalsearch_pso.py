import numpy as np
import matplotlib.pyplot as plt
import sympy as sy

from mpl_toolkits.axisartist.axislines import SubplotZero
from IPython.display import display, HTML

import matplotlib.animation as animation

def create_animation(fig, ax, result, show_annotations=True):
    num_iterations = len(result['swarm'])
    swarm_data = result['swarm'][0]
    x1 = np.array(swarm_data['x1'])
    x2 = np.array(swarm_data['x2'])
    v1 = np.array(swarm_data['v1'])
    v2 = np.array(swarm_data['v2'])

    canvas_quiver = ax.quiver([], [], [], [], width=0.005)
    canvas_points = ax.scatter([], [], color='r')
    annotations = []
    num_particles = len(result['swarm'][0]['x1'])
    for i in range(num_particles):
        annotations.append(ax.annotate(' ', (0, 0), color='#FFFFFF'))

    def _animate(k):
        swarm_data = result['swarm'][k]
        x1 = np.array(swarm_data['x1']).reshape(-1, 1)
        x2 = np.array(swarm_data['x2']).reshape(-1, 1)
        xy = np.concatenate([x1, x2], axis=1)

        canvas_points.set_offsets(xy)
        canvas_quiver.set_offsets(xy)
        canvas_quiver.set_UVC(swarm_data['v1'], swarm_data['v2'])

        for i, annot in enumerate(annotations):
            annot.set_position((x1[i], x2[i]))
            fitness = swarm_data['fitness'][i]
            best_fitness = swarm_data['best_fitness'][i]
            new_text = '[%d] %.2f (%.2f)' % (i, fitness, best_fitness) if show_annotations else ' '
            annot.set_text(new_text)
        return canvas_quiver, canvas_points, annotations

    return animation.FuncAnimation(fig, _animate, frames=num_iterations, interval=500, blit=False)

class Particle:
    def __init__(self, position, velocity, fitness=0.0):
        self._position = position
        self._velocity = velocity
        self._fitness = fitness
        self._best_position = None
        self._best_fitness = None

    def copy(self):
        return Particle(self._position, self._velocity, self._fitness)

    def position(self):
        return self._position

    def update_fitness(self, f):
        self._fitness = f.func_at(self._position)[0]
        if self._best_fitness is None or self._best_fitness > self._fitness:
            self._best_fitness = self._fitness
            self._best_position = self._position

    def update_position(self, global_best, w, c1, c2):
        r = np.random.rand(self._position.shape[0])
        s = np.random.rand(self._position.shape[0])
        inertial_term = w * self._velocity
        cognitive_term = c1 * r * (self._best_position - self._position)
        social_term = c2 * s * (global_best._position - self._position)
        # Compute new velocity and position
        self._velocity = inertial_term + cognitive_term + social_term
        self._position = self._position + self._velocity


class Swarm:
    def __init__(self, f, feasible_set, num_particles):
        self._f = f
        self._global_best = None
        self._num_particles = num_particles
        self._feasible_set = np.array(feasible_set)

        self._dimensions = self._f.dimensions() # TODO: Assume two dimensions
        self._particles = []
        self._data = []
        for i in range(num_particles):
            position = self.create_random_vector(self._feasible_set)
            velocity = self.create_random_velocity()
            self._data.append(np.array([position, velocity]))
            particle = Particle(position, velocity)
            self._particles.append(particle)

    def create_random_velocity(self):
        velocity_bounds = np.zeros((self._dimensions, 2))
        velocity_bounds[:,0] = -1  # First column represent the low
        velocity_bounds[:,1] = 1   # Second column represent the low
        return self.create_random_vector(velocity_bounds)

    def create_random_vector(self, bounds):
        """
        Creates a random vector within the feasible set.
        """
        matrix = np.array(bounds)
        low  = matrix[:,0]
        high = matrix[:,1]
        return np.random.uniform(low, high)

    def update_fitness(self):
        """
        Computes the fitness of each particles. We also find the
        individual particle's best position and the best particle
        in the swarm.
        """
        for particle in self._particles:
            particle.update_fitness(self._f)

            # Find the fittest particle in the swarm
            if self._global_best is None:
                self._global_best = particle
            # TODO: Can we use the particle's best fitness?
            if self._global_best._fitness > particle._fitness:
                self._global_best = particle.copy() # TODO: Inefficient

    def update_positions(self, w, c1, c2):
        for particle in self._particles:
            particle.update_position(self._global_best, w, c1, c2)

    def dump(self):
        ret_val = { 'x1': [], 'x2': [], 'v1': [], 'v2': [],
                   'fitness': [], 'best_fitness': [],
                   'global_best_fitness': self._global_best._fitness }
        for particle in self._particles:
            ret_val['x1'].append(particle._position[0])
            ret_val['x2'].append(particle._position[1])
            ret_val['v1'].append(particle._velocity[0])
            ret_val['v2'].append(particle._velocity[1])
            ret_val['fitness'].append(particle._fitness)
            ret_val['best_fitness'].append(particle._best_fitness)
        return ret_val

    def plot_particles(self, ax):
        x = []
        y = []
        u = []
        v = []
        fitness = []
        best_fitness = []
        for particle in self._particles:
            x.append(particle._position[0])
            y.append(particle._position[1])
            u.append(particle._velocity[0])
            v.append(particle._velocity[1])
            fitness.append(particle._fitness)
            best_fitness.append(particle._best_fitness)
        ax.scatter(x, y, c='r')
        best_pos = self._global_best._position
        ax.scatter(best_pos[0], best_pos[1], marker='h', color='#FFFF00')
        for i in range(len(x)):
            point_annotation = '[%d] %.2f (%.2f)' % (i, fitness[i], best_fitness[i])
            is_global_best = self._global_best._fitness == fitness[i]
            color = '#FFFF00' if is_global_best else '#FFFFFF'
            ax.annotate(point_annotation, (x[i], y[i]), color=color)
        ax.quiver(x, y, u, v, width=0.005)

class PSO:
    def __init__(self, f, feasible_set, num_particles):
        self._f = f
        self._num_particles = num_particles
        self._feasible_set = feasible_set

    def run(self, w=1, c1=1, c2=1, max_iterations=10):
        ret_val = { 'swarm': [], 'best_point': None, 'best_fitness': None }
        swarm = Swarm(self._f, self._feasible_set, self._num_particles)
        for k in range(1, max_iterations+1):
            swarm.update_fitness()
            ret_val['swarm'].append(swarm.dump())
            swarm.update_positions(w, c1, c2)
        ret_val['best_point'] = swarm._global_best._position
        ret_val['best_fitness'] = swarm._global_best._fitness
        return ret_val
