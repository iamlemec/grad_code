# assignment tools

import jax
import jax.lax as lax
import jax.numpy as np
from functools import partial
from valjax import smoothmax, address, interp_index, interp_address

# inject variables into object
def inject(object, params):
    for k, v in params.items():
        setattr(object, k, v)

# default algo params
alg0 = {
    'interp': False,
    'N': 500,
    'flo': 0.5,
    'fhi': 1.5,
    'ε': 1e-6,
}

class Valfunc:
    def __init__(self, par, **alg):
        inject(self, {**alg0, **alg})
        inject(self, par)

        # precompute
        self.calc_kss()
        self.make_grid()

        # optimize
        self.fast_solve = jax.jit(self.solve, static_argnames='K')

    # utility function (derivative clipped)
    def util(self, c):
        u0 = np.log(np.maximum(self.ε, c))
        u1 = np.maximum(0.0, (self.ε-c)/self.ε)
        return u0 - u1

    # production function
    def prod(self, k):
        return self.z*(k**(self.α))

    # find steady state
    def calc_kss(self):
        rhs = (1-self.β)/self.β + self.δ
        self.kss = (self.α*self.z/rhs)**(1/(1-self.α))

    # grid and steady state
    def make_grid(self):
        self.klo, self.khi = self.flo*self.kss, self.fhi*self.kss
        self.k_grid = np.linspace(self.klo, self.khi, self.N)
        self.y_grid = self.prod(self.k_grid)
        self.yp_grid = self.y_grid + (1-self.δ)*self.k_grid
        self.cp_grid = self.yp_grid[:,None] - self.k_grid[None,:]
        self.up_grid = self.util(self.cp_grid)

    # compute full bellman update
    def update_step(self, v0):
        vp = self.up_grid + self.β*v0[None,:]
        if self.interp:
            ik = smoothmax(vp, axis=1)
            v1 = interp_address(vp, ik, axis=1)
            k1 = interp_index(self.k_grid, ik)
        else:
            ik = np.argmax(vp, axis=1)
            v1 = address(vp, ik, axis=1)
            k1 = self.k_grid[ik]
        return v1, (v1, k1)

    # solve value function
    def solve(self, K=1000):
        v0 = self.util(self.y_grid-self.δ*self.k_grid)
        upd = lambda s, t: self.update_step(s)
        v1, hist = lax.scan(upd, v0, np.arange(K))
        _, (_, k1) = self.update_step(v1)
        return v1, k1, hist
