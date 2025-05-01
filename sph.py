# Tutorial Exercise 3
# Initial implementation for 1D Smoothed Particle Hydrodynamics solver
#
# authors:
# - Paul Römer           7377945
# - Philip Julius Pupkes 7360318
# - Alice Coors          7392745
# - Muhammad Fakhar      7432447

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

    def __init__(self, param: Parameters, file, m: np.array, r: np.array, v: np.array, u: np.array):
        """
        initialize the state and set the initial conditions and free parameters of the system
            file: file to write the results to
        m: masses the particles
        r: initial positions of the particles
        v: initial velocities of the particles
        u: initial internal energies of the particles
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
        self.u = u
        self.du = np.zeros(self.N)
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

    def neighbors_of(self, i):
        """
        return the indices of all neighbors of particle i.
        a j particle is a neighbor of particle i if |r_i - r_j| <= 2h (including the particle itself)
        """
        # find all indices in self.r for which |r_i - r_j| <= 2h
        return np.where(np.abs(self.r[i] - self.r) <= 2*self.h)[0]

    def compute_density(self):
        """compute the current density at each particle position"""
        for i in range(self.N):
            neighbors = self.neighbors_of(i)
            s = np.abs(self.r[i] - self.r[neighbors]) / self.h
            self.rho[i] = np.sum(self.m[neighbors] * W(s, self.h))

    def compute_pressure(self):
        """compute the pressure at each particle position from the current density using an isothermal ideal gas law"""
        self.P[:] = (self.param.gamma - 1) * self.rho * self.u

    def compute_forces(self):
        """compute the forces -> accelerations acting on each particle from the current pressure, density and particle positions (as well as particle masses)"""
        for i in range(self.N):
            neighbors = self.neighbors_of(i)
            dP = (self.P[i]/self.rho[i]**2) + self.P[neighbors]/self.rho[neighbors]**2
            r_ij = self.r[i] - self.r[neighbors]
            s = np.abs(r_ij) / self.h
            self.a[i] = -np.sum(self.m[neighbors] * dP * dW(s, self.h) * np.sign(self.r[neighbors]))

    def compute_energy_flux(self):
        for i in range(self.N):
            neighbors = self.neighbors_of(i)
            r_ij = self.r[i] - self.r[neighbors]
            s = np.abs(r_ij) / self.h
            dv = self.v[i] - self.v[neighbors]
            self.du[i] = self.P[i]/self.rho[i]**2 * np.sum(self.m[neighbors] * dv * dW(s, self.h) * np.sign(self.r[neighbors]))

    def determine_timestep(self):
        """
        determine the global timestep to use based on the minimum timestep necessary to fullfill a CFL
        condition for every particle
        """

        epsilon = 1e-8

        dv = np.zeros_like(self.v)
        for i in range(self.N):
            neighbors = self.neighbors_of(i)
            s = np.abs(self.r[i] - self.r[neighbors]) / self.h
            dv[i] = np.sum(self.m[neighbors] / self.rho[neighbors] * self.v[neighbors] * dW(s, self.h))

        a = self.h / (self.h * np.abs(dv) + self.param.c_s)
        b = np.sqrt(self.h / (np.abs(self.a) + epsilon))
        self.dt = self.param.cfl * np.min(np.minimum(a, b))

    def forward_euler_timestep(self):
        """
        perform an explicit euler timestep to advance the positions and velocities of all particles
        based on the current computed accelerations `a`
        """

        self.r += self.v * self.dt
        self.v += self.a * self.dt
        self.u += self.du * self.dt

    def write_to_file(self, time_step, time):
        """write the current positions and velocities to the output file f (if there is one)"""

        if self.file is not None:

            if not self.wrote_header:
                self.file.write("time_step,h,dt,time,index,position,velocity,density,energy\n")
                self.wrote_header = True;

            for i in range(self.N):
                self.file.write(f"{time_step},{self.h},{self.dt},{time},{i},{self.r[i]},{self.v[i]},{self.rho[i]},{self.u[i]}\n")


def initial_state(file, N, eta, cfl):

    gamma = 5/3
    c_s = 1.0

    p = np.linspace(0.0, 1.0, N+2)[1:-1]
    assert len(p) == N
    v = np.zeros(N)
    m = np.ones(N) / N
    u = np.ones(N) * np.sqrt(c_s)

    params = Parameters(gamma, c_s, eta, cfl)
    state = State(params, file, m, p, v, u)

    return state


def main(file, N, eta, cfl):
    """perform the computation, optionally write the result to the passed in file `file`"""

    state = initial_state(file, N, eta, cfl)

    T = 1.0
    write_interval = 0.01

    # perfom simulations steps until the total time elapsed has reached the desired end time
    t = 0.0
    time_step = 0
    next_write = 0.0
    last_write = -1
    while True:
        print(f"\rperforming timestep {time_step:6d} for t/T={t/T:.2%}", end='')
        sys.stdout.flush()
        state.compute_smoothing_length()
        state.compute_density()
        state.compute_pressure()
        state.compute_forces()
        state.compute_energy_flux()
        state.determine_timestep()
        state.forward_euler_timestep()

        if t >= next_write:
            state.write_to_file(time_step, t)
            next_write += write_interval
            last_write = time_step

        if t >= T:
            break

        t += state.dt
        time_step += 1

    # if we did not just write, write the final state
    if last_write != time_step-1:
         state.write_to_file(time_step-1, t-state.dt)

    print()

if __name__ == "__main__":

    N = int(sys.argv[1])
    eta = float(sys.argv[2])
    cfl = float(sys.argv[3])

    with open(f"results/results_{N}_{eta}_{cfl}.csv", "w") as f:
        main(f, N, eta, cfl)
