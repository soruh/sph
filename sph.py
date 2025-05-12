# Tutorial Exercise 4
# Initial implementation for 1D Smoothed Particle Hydrodynamics solver
#
# authors:
# - Paul Römer           7377945
# - Philip Julius Pupkes 7360318
# - Alice Coors          7392745
# - Muhammad Fakhar      7432447

import numpy as np
import sys

epsilon = 1e-12


# numpy vectorized implemenation of the smoothing kernel
def W(s, h):
    s = np.asarray(s)
    result = np.zeros_like(s)

    mask1 = (s >= 0) & (s <= 1)
    mask2 = (s > 1) & (s <= 2)

    result[mask1] = 1 - 1.5 * s[mask1] ** 2 + 0.75 * s[mask1] ** 3
    result[mask2] = 0.25 * (2 - s[mask2]) ** 3

    return (2 / (3 * h)) * result


# numpy vectorized implemenation of the smoothing kernel derivative
def dW(s, h):
    s = np.asarray(s)
    result = np.zeros_like(s)

    mask1 = (s >= 0) & (s <= 1)
    mask2 = (s > 1) & (s <= 2)

    result[mask1] = -3 * s[mask1] + (9 / 4) * s[mask1] ** 2
    result[mask2] = -0.75 * (2 - s[mask2]) ** 2

    return (2 / (3 * h**2)) * result


class Parameters:
    """free parameters of the SPH simulation"""

    def __init__(self, gamma, u0, eta, cfl, alpha):
        """
        gamma: specific heat ratio
         u0: initial velocity
        eta: order of unity
        cfl: time step size as a fraction of the maximum stable timestep
        """

        self.gamma = gamma
        self.u0 = u0
        self.eta = eta
        self.cfl = cfl
        self.alpha = alpha
        self.beta = 2 * alpha


class State:
    """structure storing the simulation state"""

    def __init__(
        self,
        param: Parameters,
        file,
        m: np.array,
        r: np.array,
        v: np.array,
        u: np.array,
    ):
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
        self.rho = np.zeros(self.N)
        self.a = np.zeros(self.N)
        self.du = np.zeros(self.N)
        self.c_s = np.zeros(self.N)
        self.P = np.zeros(self.N)

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
        self.h = self.param.eta * np.mean(min_dists)

    def neighbors_of(self, i):
        """
        return the indices of all neighbors of particle i.
        a j particle is a neighbor of particle i if |r_i - r_j| <= 2h (including the particle itself)
        """
        # find all indices in self.r for which |r_i - r_j| <= 2h
        return np.where(np.abs(self.r[i] - self.r) <= 2 * self.h)[0]

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
            dP = (self.P[i] / self.rho[i] ** 2) + self.P[neighbors] / self.rho[
                neighbors
            ] ** 2
            r_ij = self.r[i] - self.r[neighbors]
            s = np.abs(r_ij) / self.h
            self.a[i] = -np.sum(self.m[neighbors] * dP * dW(s, self.h) * np.sign(r_ij))

    def compute_energy_flux(self):
        for i in range(self.N):
            neighbors = self.neighbors_of(i)
            r_ij = self.r[i] - self.r[neighbors]
            s = np.abs(r_ij) / self.h
            dv = self.v[i] - self.v[neighbors]
            self.du[i] = (
                self.P[i]
                / self.rho[i] ** 2
                * np.sum(self.m[neighbors] * dv * dW(s, self.h) * np.sign(r_ij))
            )

    def compute_speed_of_sound(self):
        self.c_s[:] = np.sqrt(self.param.gamma * (self.param.gamma - 1) * self.u)

    def determine_timestep(self):
        """
        determine the global timestep to use based on the minimum timestep necessary to fullfill a CFL
        condition for every particle
        """

        # compute velocity divergence
        dv = np.zeros_like(self.v)
        for i in range(self.N):
            neighbors = self.neighbors_of(i)
            r_ij = self.r[i] - self.r[neighbors]
            s = np.abs(r_ij) / self.h
            dv[i] = np.sum(
                self.m[neighbors]
                / self.rho[neighbors]
                * self.v[neighbors]
                * dW(s, self.h)
                * np.sign(r_ij)
            )

        dv = np.abs(dv)

        a = self.h / (self.h * dv + self.c_s)
        b = np.sqrt(self.h / (np.abs(self.a) + epsilon))
        c = self.h / (
            (1.0 + 1.2 * self.param.alpha) * self.c_s  #
            + (1.0 + 1.2 * self.param.beta) * self.h * dv
        )
        self.dt = self.param.cfl * np.min(np.minimum(a, b, c))

    def artificial_viscosity(self):
        for i in range(self.N):
            neighbors = self.neighbors_of(i)

            r_ij = self.r[i] - self.r[neighbors]
            v_ij = self.v[i] - self.v[neighbors]
            vr_ij = v_ij * r_ij
            mask = np.where(vr_ij < 0.0)[0]  # not <= to prevent divide by zero
            full_mask = neighbors[mask]

            r_ij = r_ij[mask]
            v_ij = v_ij[mask]
            vr_ij = vr_ij[mask]

            rho_ij = (self.rho[i] + self.rho[full_mask]) / 2.0
            v_sig = (
                self.c_s[i]
                - self.c_s[full_mask]
                - self.param.beta * vr_ij / np.abs(r_ij)
            )

            Pi = (self.param.alpha * v_sig * vr_ij) / (2.0 * rho_ij * r_ij)
            Lambda = (self.param.alpha * v_sig * vr_ij**2) / (2.0 * rho_ij * r_ij**2)

            s = np.abs(r_ij) / self.h
            m = self.m[full_mask]

            self.a[i] += np.sum(m * Pi * dW(s, self.h))
            self.du[i] += np.sum(m * Lambda * dW(s, self.h) * np.sign(r_ij))

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
                self.file.write(
                    "time_step,h,dt,time,index,position,velocity,density,energy,pressure\n"
                )
                self.wrote_header = True

            for i in range(self.N):
                self.file.write(
                    f"{time_step},{self.h},{self.dt},{time},{i},{self.r[i]},{self.v[i]},{self.rho[i]},{self.u[i]},{self.P[i]}\n"
                )


def setup_homogenous(file, N, eta, cfl):

    gamma = 5 / 3
    u0 = 1.0

    r = np.linspace(0.0, 1.0, N + 2)[1:-1]
    assert len(r) == N
    v = np.zeros(N)
    m = np.ones(N) / N
    u = np.ones(N) * u0

    params = Parameters(gamma, u0, eta, cfl, 1.0)
    state = State(params, file, m, r, v, u)

    return state


def setup_sod_shock(file, N, alpha):

    x_min, x_max = -0.5, 1.5
    x_shock = 0.5

    eta = 2
    cfl = 0.1
    gamma = 1.4
    u0 = 1.0

    # space particles evenly
    x = np.linspace(x_min, x_max, N)
    dx = x[1] - x[0]

    mask_pre = x <= x_shock
    mask_post = x > x_shock

    # target values
    P = np.zeros(N)
    P[mask_pre] = 1.0
    P[mask_post] = 0.1

    rho = np.zeros(N)
    rho[mask_pre] = 1.0
    rho[mask_post] = 0.125

    v = np.zeros(N)

    # compute initial conditions to reach target values
    u = P / ((gamma - 1) * rho)
    m = rho * dx

    params = Parameters(gamma, u0, eta, cfl, alpha)
    state = State(params, file, m, x, v, u)

    return state


def main(file, N, alpha):
    """perform the computation, optionally write the result to the passed in file `file`"""

    # state = setup_homogenous(file, N, eta, cfl)
    state = setup_sod_shock(file, N, alpha)

    T = 0.2
    write_interval = 0.001

    # perfom simulations steps until the total time elapsed has reached the desired end time
    t = 0.0
    time_step = 0
    next_write = 0.0
    last_write = -1
    while True:
        print(f"\rperforming timestep {time_step:6d} for t/T={t/T:.2%}", end="")
        sys.stdout.flush()
        state.compute_smoothing_length()
        state.compute_density()
        state.compute_pressure()
        state.compute_forces()
        state.compute_energy_flux()
        state.compute_speed_of_sound()
        state.artificial_viscosity()
        state.determine_timestep()

        if time_step == 0:
            state.write_to_file(time_step, t)

        state.forward_euler_timestep()

        t += state.dt
        time_step += 1

        done = state.dt <= 0 or t >= T or t < 0 or np.isnan(t)

        if t >= next_write or done:
            state.write_to_file(time_step, t)
            next_write += write_interval
            last_write = time_step

        if done:
            break

    print()


if __name__ == "__main__":

    N = int(sys.argv[1])
    alpha = float(sys.argv[2])

    with open(f"results/shock_{N}_{alpha}.csv", "w") as f:
        main(f, N, alpha)
