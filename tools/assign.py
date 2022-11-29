# assignment tools

import jax
import jax.lax as lax
import jax.numpy as np
from functools import partial
from valjax import optim_secant

# inject variables into object
def inject(object, params):
    for k, v in params.items():
        setattr(object, k, v)

class Valfunc:
    def __init__(self, par, alg):
        inject(self, par)
        inject(self, alg)

        # precompute
        self.calc_kss()
        self.make_grid()

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

class ValfuncGrid(Valfunc):
    def __init__(self, par, alg):
        super().__init__(par, alg)

        # optimize
        self.j_solve = jax.jit(self.solve)

    def make_grid(self):
        super().make_grid()

        # cross grid for consumption
        self.cp_grid = self.yp_grid[:,None] - self.k_grid[None,:]

    # bellman equation evaluate
    def eval_policy(self, vp):
        return self.util(self.cp_grid) + self.β*vp[None,:]

    # update using grid
    def update_step(self, v0):
        v1 = self.eval_policy(v0)
        return np.max(v1, axis=1)

    # execute interpolation solve
    def solve(self, K=300):
        # initial guesses
        c0 = self.y_grid - self.δ*self.k_grid
        v = self.util(c0)

        # iterate on update
        upd = lambda i, vi: self.update_step(vi)
        v = lax.fori_loop(0, K, upd, v)

        # get optimal policy
        vk = self.eval_policy(v)
        ik = np.argmax(vk, axis=1)
        kp = self.k_grid[ik]

        return v, kp

class ValfuncInterp(Valfunc):
    def __init__(self, par, alg):
        super().__init__(par, alg)

        # vectorize
        self.d_bellman_interp = jax.grad(self.bellman_interp, argnums=0)
        self.v_bellman_interp = jax.vmap(self.bellman_interp, in_axes=(0, 0, None))
        self.vd_bellman_interp = jax.vmap(self.d_bellman_interp, in_axes=(0, 0, None))

        # optimize
        self.j_update_step = jax.jit(self.update_step)

    # bellman equation evaluate
    def bellman(self, kp, yp, vp):
        kp1 = np.maximum(0.0, kp)
        return self.util(yp-kp1) + self.β*vp

    # interpolated bellman calculation
    def bellman_interp(self, kp, yp, v):
        vp = np.interp(kp, self.k_grid, v)
        return self.bellman(kp, yp, vp)

    # update using intpolation
    def update_step(self, k0, v0):
        # update policy
        dvdk = self.vd_bellman_interp(k0, self.yp_grid, v0)
        dvdk1 = np.clip(dvdk, -1.0, 1.0)
        k1 = np.clip(k0 + self.Δk*dvdk1, self.klo, self.khi)

        # update value
        vx = self.bellman(k0, self.yp_grid, v0)
        v1 = np.clip(self.Δv*vx + (1-self.Δv)*v0, self.vlo, self.vhi)

        return k1, v1

    # execute interpolation solve
    def solve(self, K=3000, per=1000):
        # initial guesses
        k = self.k_grid
        v = self.util(self.y_grid-self.δ*self.k_grid)

        # iterate on update
        for i in range(K+1):
            kp, vp = self.j_update_step(k, v)

            err_k = np.max(np.abs(kp-k))
            err_v = np.max(np.abs(vp-v))
            err = np.maximum(err_k, err_v)

            if i == 0 or i == K or i % per == 0:
                print(i, err_k, err_v)

            k, v = kp, vp

        return kp, vp
