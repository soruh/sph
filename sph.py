# Tutorial Exercise 1
# Initial layout for Smoothed Particle Hydrodynamics solver
#
# author: Paul Römer 7377945

import numpy as np


def W(s):
    """smoothing kernel W"""
    # use M4 kernel as shown in instructuions
    raise NotImplementedError("kernel W(s)")


def dW(s):
    """gradient of smoothing kernel W"""
    # use M4 kernel as shown in instructuions
    raise NotImplementedError("kernel derivative dW(s)")


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

        self.m = m
        self.r = r
        self.v = v
        self.a = np.zeros(self.N)
        self.P = np.zeros(self.N)
        self.rho = np.zeros(self.N)

    @property
    def N(self):
        return len(self.p)

    def compute_smoothing_length(self):
        """compute a global smoothing length based on the average spacing"""
        # eta * mean(min(|r_j - ri|))
        raise NotImplementedError("compute_smoothing_length")

    def neightbors_of(self, i):
        """
        return the indices of all neighbors of particle i.
        a j particle is a neighbor of particle i if |r_i - r_j| <= 2h (including the particle itself)
        """
        # find all indices in self.r for which |r_i - r_j| <= 2h
        raise NotImplementedError("neightbors_of")

    def compute_density_i(self, i):
        """compute the current at the position of particle i"""

        # compute density by integrating mass over W(neightbors_of(i))
        raise NotImplementedError("compute_density_i")

    def compute_density(self):
        """compute the current density at each particle position"""

        for i in range(0, self.N):
            self.rho[i] = self.compute_density_i(i)

    def compute_pressure(self):
        """compute the pressure at each particle position from the current density using an isothermal ideal gas law"""

        self.P[:] = (self.param.gamma - 1) * self.rho * self.param.u0

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
        self.dt = min([self.max_timestep_i(i) for i in range(0, self.N)])

    def forward_euler_timestep(self):
        """
        perform an explicit euler timestep to advance the positions and velocities of all particles
        based on the current computed accelerations `a`
        """

        self.r += self.v * self.dt
        self.v += self.a * self.dt

    def write_to_file(self):
        """write the current positions and velocities to the output file f (if there is one)"""

        if self.f is not None:
            # write once csv row containing 2N entries, first the positions, then the velocities
            raise NotImplementedError("write_to_file")


def main(file=None):
    """perform the computation, optionally write the result to the passed in file `file`"""

    # todo: initial conditions
    # T = ...
    # params = Param(gamma, c_s, eta, cfl)
    # state = State(params, file, m, p, v)

    state: State = None
    T = 10

    # perfom simulations steps until the total time elapsed has reached the desired end time
    t = 0.0
    i = 0
    while t < T:
        print(f"performing timestep {i:6d} for t/T={(t/T * 100.0):.2%}")
        state.compute_smoothing_length()
        state.compute_density()
        state.compute_pressure()
        state.compute_forces()
        state.determine_timestep()
        state.forward_euler_timestep()
        state.write_to_file()

        t += state.dt
        i += 1


if __name__ == "__main__":
    with open("results.csv", "w") as f:
        main(f)
