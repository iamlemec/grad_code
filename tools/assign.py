# assignment tools

import jax
import jax.numpy as np
from functools import partial

# inject variables into object
def inject(object, params):
    for k, v in params.items():
        setattr(object, k, v)

class ValfuncInterp:
    def __init__(self, par, alg):
        inject(self, par)
        inject(self, alg)

        # precompute
        self.calc_kss()
        self.make_grid()

        # vectorize
        self.d_bellman_interp = jax.grad(self.bellman_interp, argnums=0)
        self.v_bellman_interp = jax.vmap(self.bellman_interp, in_axes=(0, 0, None))
        self.vd_bellman_interp = jax.vmap(self.d_bellman_interp, in_axes=(0, 0, None))

        # optimize
        self.j_update_step = jax.jit(self.update_step)

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

    # bellman equation evaluate
    def bellman(self, kp, yp, vp):
        kp1 = np.maximum(0.0, kp)
        return self.util(yp-kp1) + self.β*vp

    # interpolated bellman calculation
    def bellman_interp(self, kp, yp, v):
        vp = np.interp(kp, self.k_grid, v)
        return self.bellman(kp, yp, vp)

    # update using intpolation
    def update_step(self, kp, vp):
        # update policy
        dvdk = self.vd_bellman_interp(kp, self.yp_grid, vp)
        dvdk1 = np.clip(dvdk, -1.0, 1.0)
        k1 = np.clip(kp + self.Δk*dvdk1, self.klo, self.khi)

        # update value
        for i in range(10):
            vx = self.bellman(kp, self.yp_grid, vp)
            vp = self.Δv*vx + (1-self.Δv)*vp

        return k1, vp

    # execute interpolation solve
    def solve_interp(self, K=3000, per=1000):
        # initial guesses
        k = self.k_grid
        v = self.util(self.y_grid-self.δ*self.k_grid)

        # iterate on update
        for i in range(K):
            kp, vp = self.j_update_step(k, v)

            err_k = np.max(np.abs(kp-k))
            err_v = np.max(np.abs(vp-v))
            err = np.maximum(err_k, err_v)

            if i == 0 or i == K - 1 or i % per == 0:
                print(i, err_k, err_v)

            k, v = kp, vp

        return kp, vp
