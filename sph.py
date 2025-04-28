# Tutorial Exercise 3
# Initial implementation for 1D Smoothed Particle Hydrodynamics solver
#
# author: Paul Römer 7377945

import numpy as np
import sys


# def W(h, s):
#     """smoothing kernel W"""
#     # use M4 kernel as shown in instructuions
#     if 0 <= s and s <= 1:
#         return 2/(3 * h) * (1 - 3/2 * s**2 + 3/4 * s**3)
#     elif 1 <= s and s <= 2:
#         return 2/(12 * h) * (2 - s)**3
#     else:
#         return 0


# def dW(h, s):
#     """gradient of smoothing kernel W"""
#     # use M4 kernel as shown in instructuions
#     if 0 <= s and s <= 1:
#         return 2/h**2 * (-s + 3/4 * s**2)
#     elif 1 <= s and s <= 2:
#         return -1/(2 * h**2) * (2 - s)**2
#     else:
#         return 0


# numpy vectorized implemenation of the smoothing kernel
def W(s, h):
    s = np.asarray(s)
    result = np.zeros_like(s)

    mask1 = (s >= 0) & (s <= 1)
    mask2 = (s > 1) & (s <= 2)

    result[mask1] = 1 - 1.5 * s[mask1]**2 + 0.75 * s[mask1]**3
    result[mask2] = 0.25 * (2 - s[mask2])**3

    return (2 / (3 * h)) * result

# numpy vectorized implemenation of the smoothing kernel derivative
def dW(s, h):
    s = np.asarray(s)
    result = np.zeros_like(s)

    mask1 = (s >= 0) & (s <= 1)
    mask2 = (s > 1) & (s <= 2)

    result[mask1] = -3 * s[mask1] + (9/4) * s[mask1]**2
    result[mask2] = -0.75 * (2 - s[mask2])**2

    return (2 / (3 * h**2)) * result




class Parameters:
    """free parameters of the SPH simulation"""

    def __init__(self, gamma, c_s, eta, cfl):
        """
        gamma: specific heat ratio
        c_s: speed of sound
        eta: order of unity
        cfl: time step size as a fraction of the maximum stable timestep
        """

        self.gamma = gamma
        self.c_s = c_s
        self.u0 = c_s**2
        self.eta = eta
        self.cfl = cfl


class State:
    """structure storing the simulation state"""

    def __init__(self, param: Parameters, file, m: np.array, r: np.array, v: np.array):
        """
        initialize the state and set the initial conditions and free parameters of the system
            file: file to write the results to
        m: masses the particles
        r: initial positions the particles
        v: initial velocities the particles
        params: free parameters of the simulation
        """

        assert isinstance(m, np.ndarray)
        assert isinstance(r, np.ndarray)
        assert isinstance(v, np.ndarray)

        assert len(m) == len(r)
        assert len(v) == len(r)

        assert isinstance(param, Parameters)
        assert file is None or hasattr(f, "write")

        self.param = param

        self.h = None
        self.dt = None

        self.file = file
        self.wrote_header = False

        self.m = m
        self.r = r
        self.v = v
        self.a = np.zeros(self.N)
        self.P = np.zeros(self.N)
        self.rho = np.zeros(self.N)

    @property
    def N(self):
        return len(self.r)

    def compute_smoothing_length(self):
        """compute a global smoothing length based on the average spacing"""

        # sort positions for cheaper closest distance check
        r_sorted = np.sort(self.r)
        diff = np.abs(np.diff(r_sorted))

        # minimum neighbor distance of each particle
        min_dists = np.empty_like(r_sorted)
        min_dists[1:-1] = np.minimum(diff[:-1], diff[1:])
        min_dists[0] = diff[0]
        min_dists[-1] = diff[-1]

        # mean minimum neighbor distance
        self.h = eta * np.mean(min_dists)

    def neightbors_of(self, i):
        """
        return the indices of all neighbors of particle i.
        a j particle is a neighbor of particle i if |r_i - r_j| <= 2h (including the particle itself)
        """
        # find all indices in self.r for which |r_i - r_j| <= 2h
        return np.where(np.abs(self.r[i] - self.r) <= 2*self.h)[0]

    def compute_density_i(self, i):
        """compute the current density at the position of particle i"""
        neighbors = self.neightbors_of(i)
        s = np.abs(self.r[i] - self.r[neighbors]) / self.h
        return np.sum(self.m[neighbors] * W(s, self.h))

    def compute_density(self):
        """compute the current density at each particle position"""

        for i in range(0, self.N):
            self.rho[i] = self.compute_density_i(i)

    def compute_pressure(self):
        """compute the pressure at each particle position from the current density using an isothermal ideal gas law"""

        # self.P[:] = (self.param.gamma - 1) * self.rho * self.v
        self.P[:] = self.rho * self.v


    def compute_forces(self):
        """compute the forces -> accelerations acting on each particle from the current pressure, density and particle positions (as well as particle masses)"""

        # compute self.a by integrating - m_j * (P_i / rho_i^2 + P_j / rho_j^2) over dW(neightbors_of(i))
        raise NotImplementedError("compute_forces")

    def max_timestep_i(self, i):
        """determine the local timestep that would fullfill a CFL condition for particle i"""

        # self.cfl * min(h/(h * grad(v)_i) + c), sqrt(h/(abs(a) + c)))
        raise NotImplementedError("max_timestep_i")

    def determine_timestep(self):
        """
        determine the global timestep to use based on the minimum timestep necessary to fullfill a CFL
        condition for every particle
        """

        # determine timestep as min(\Delta T_{max,i})
        self.dt = min([self.max_timestep_i(time_step) for time_step in range(self.N)])

    def forward_euler_timestep(self):
        """
        perform an explicit euler timestep to advance the positions and velocities of all particles
        based on the current computed accelerations `a`
        """

        self.r += self.v * self.dt
        self.v += self.a * self.dt

    def write_to_file(self, time_step, time):
        """write the current positions and velocities to the output file f (if there is one)"""

        if self.file is not None:

            if not self.wrote_header:
                self.file.write(f"time_step,time,index,position,velocity,density\n")
                self.wrote_header = True;

            for i in range(self.N):
                self.file.write(f"{time_step},{time},{i},{self.r[i]},{self.v[i]},{self.rho[i]}\n")


def initial_state(file, N, eta):

    p = np.linspace(0.05, 0.95, N)
    v = np.zeros(N)
    m = np.ones(N)

    gamma = 1.0
    c_s = 1.0
    cfl = 1.0

    params = Parameters(gamma, c_s, eta, cfl)
    state = State(params, file, m, p, v)

    return state


def main(file, N, eta):
    """perform the computation, optionally write the result to the passed in file `file`"""

    state = initial_state(file, N, eta)

    # temporary values as we won't do any time steps
    T = 1.0
    state.dt = 0.0

    # perfom simulations steps until the total time elapsed has reached the desired end time
    t = 0.0
    time_step = 0
    # while t < T:
    while time_step < 1:
        print(f"performing timestep {time_step:6d} for t/T={(t/T * 100.0):.2%}")
        state.compute_smoothing_length()
        state.compute_density()
        # state.compute_pressure()
        # state.compute_forces()
        # state.determine_timestep()
        # state.forward_euler_timestep()
        state.write_to_file(time_step, t)

        t += state.dt
        time_step += 1


if __name__ == "__main__":

    N = int(sys.argv[1])
    eta = float(sys.argv[2])

    with open(f"results/results_{N}_{eta}.csv", "w") as f:
        main(f, N, eta)
