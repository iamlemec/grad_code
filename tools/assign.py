# assignment tools

import jax
import jax.lax as lax
import jax.numpy as np
import jax.scipy as sp

from valjax import smoothmax, address, interp_index, interp_address
from .models import normed, split_range_vec, null_space

# default algo params
alg0 = {
    'N': 1000,
    'flo': 0.2,
    'fhi': 1.5,
    'ε': 1e-6,
    'interp': False,
}

class Capital:
    def __init__(self, par, **alg):
        # store parameters (somewhat hacky)
        self.__dict__.update({**par, **alg0, **alg})

        # precompute
        self.calc_kss()
        self.make_grid()

        # optimize
        self.fast_bellman = jax.jit(self.solve_bellman, static_argnames='K')
        self.fast_distribution = jax.jit(self.solve_distribution, static_argnames=['method', 'T'])

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

    # grid and precompute
    def make_grid(self):
        # construct uniform grid
        self.klo, self.khi = self.flo*self.kss, self.fhi*self.kss
        self.k_bins = np.linspace(self.klo, self.khi, self.N+1)
        self.k_grid = 0.5*(self.k_bins[:-1]+self.k_bins[1:])
        self.k_size = np.diff(self.k_bins)

        # state-choice values
        self.y_grid = self.prod(self.k_grid)
        yp_grid = self.y_grid + (1-self.δ)*self.k_grid
        cp_grid = yp_grid[:,None] - self.k_grid[None,:]
        self.up_grid = self.util(cp_grid)

        # entry distribution
        zk_bins = (np.log(self.k_bins)-self.μ)/self.σ
        ent_cdf = sp.stats.norm.cdf(zk_bins)
        self.ent_pmf = normed(np.diff(ent_cdf))

    # compute full bellman update
    def update_bellman(self, v0):
        # compute objective values
        vp = self.up_grid + self.β*v0[None,:]

        # find polict and max
        if self.interp:
            ik = smoothmax(vp, axis=1)
            v1 = interp_address(vp, ik, axis=1)
            k1 = interp_index(self.k_grid, ik)
        else:
            ik = np.argmax(vp, axis=1)
            v1 = address(vp, ik, axis=1)
            k1 = self.k_grid[ik]

        # compute statistics
        err = np.max(np.abs(v1-v0))
        ret = {'v': v1, 'kp': k1, 'err': err}

        return v1, ret

    # solve value function
    def solve_bellman(self, K=1000):
        # initial guess
        v0 = self.util(self.y_grid-self.δ*self.k_grid)

        # fixed length iteration
        upd = lambda s, t: self.update_bellman(s)
        v1, hist = lax.scan(upd, v0, np.arange(K))
        _, ret = self.update_bellman(v1)

        return v1, ret, hist

    # make conditional and unconditional transitions matrices
    def make_tmats(self, kp):
        tmat0 = split_range_vec(
            kp-0.5*self.k_size, kp+0.5*self.k_size, self.k_bins
        )/self.k_size
        tmat = self.κ*self.ent_pmf[None,:] + (1-self.κ)*tmat0
        return normed(tmat0, axis=1), normed(tmat, axis=1)

    # solve for steady state k distribution
    def solve_distribution(self, kp, method='null', T=50):
        tmat0, tmat = self.make_tmats(kp)

        if method == 'null':
            tmat1 = tmat.T - np.eye(self.N)
            _, _, vh = np.linalg.svd(tmat1, full_matrices=True)
            k_dist = vh[-1,:] # hope for the best!
        elif method == 'age':
            age_pmf = self.κ*(1-self.κ)**np.arange(T)
            upd = lambda x, t: (x @ tmat0, x)
            _, k_hist = lax.scan(upd, self.ent_pmf, np.arange(T))
            k_dist = np.sum(k_hist*age_pmf[:,None], axis=0)

        return normed(k_dist)
