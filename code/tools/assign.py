# assignment tools

import jax
import jax.lax as lax
import jax.numpy as np
import jax.scipy as sp

import valjax as vj
from .models import normed, split_range_vec

# default algo params
alg0 = {
    'N': 1000, # grid size
    'P': 10, # gradient steps
    'Δ': 0.01, # step size
    'flo': 0.2, # fraction grid low
    'fhi': 1.5, # fraction grid high
    'ε': 1e-6, # utility clip
    'smooth': False, # use smoothmax
}

# default params
par0 = {
    'γ': 0.0, # distribution noise
}

class Capital:
    def __init__(self, par, **alg):
        self.__dict__.update({**par0, **par, **alg0, **alg})
        self.calc_kss()
        self.make_grid()
        self.compile_funcs()

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
        self.yd_grid = self.y_grid + (1-self.δ)*self.k_grid
        cp_grid = self.yd_grid[:,None] - self.k_grid[None,:]
        self.up_grid = self.util(cp_grid)

        # entry distribution
        zk_bins = (np.log(self.k_bins)-self.μ)/self.σ
        ent_cdf = sp.stats.norm.cdf(zk_bins)
        self.ent_pmf = normed(np.diff(ent_cdf))

        # diffusion distribution
        if self.γ > 0.0:
            k_diff = self.k_bins[None,1:-1] - self.k_grid[:,None]
            self.dmat = np.diff(np.hstack([
                np.zeros(self.N)[:,None],
                sp.stats.norm.cdf(k_diff/self.γ),
                np.ones(self.N)[:,None],
            ]), axis=1)
        else:
            self.dmat = np.eye(self.N)

    # vmap and jit functions
    def compile_funcs(self):
        self.spec = vj.Spec(vj.SpecRange(self.klo, self.khi))
        self.l_bellman = self.spec.decoder(self.bellman, arg='kp')
        self.dl_bellman = jax.grad(self.l_bellman, argnums=0)
        self.v_bellman = jax.vmap(self.bellman, in_axes=(0, 0, None))
        self.v_policy = jax.vmap(self.policy, in_axes=(0, 0, None))

        # optimize
        self.fast_solve_grid = jax.jit(self.solve_grid, static_argnames='K')
        self.fast_solve_interp = jax.jit(self.solve_interp, static_argnames='K')
        self.fast_distribution = jax.jit(self.solve_distribution, static_argnames=['method', 'T'])

    # utility function (smooth clipped)
    def util(self, c):
        u0 = np.log(self.ε) + (c/self.ε-1)
        u1 = np.log(np.maximum(self.ε, c))
        return np.where(c >= self.ε, u1, u0)

    # production function
    def prod(self, k):
        return self.z*(k**self.α)

    # compute full bellman update
    def update_grid(self, v0):
        # compute objective values
        vp = self.up_grid + self.β*v0[None,:]

        # find polict and max
        if self.smooth:
            ik = vj.smoothmax(vp, axis=1)
            v1 = vj.interp_address(vp, ik, axis=1)
            k1 = vj.interp_index(self.k_grid, ik)
        else:
            ik = np.argmax(vp, axis=1)
            v1 = vj.address(vp, ik, axis=1)
            k1 = self.k_grid[ik]

        # compute statistics
        err = np.max(np.abs(v1-v0))
        ret = {'v': v1, 'kp': k1, 'err': err}

        return v1, ret

    # solve value function
    def solve_grid(self, K=400):
        # initial guess
        v0 = self.util(self.y_grid-self.δ*self.k_grid)

        # fixed length iteration
        upd = lambda s, t: self.update_grid(s)
        v1, hist = lax.scan(upd, v0, np.arange(K))
        _, ret = self.update_grid(v1)

        return v1, ret, hist

    # compute bellman update
    def bellman(self, kp, yd, v):
        vp = np.interp(kp, self.k_grid, v)
        up = self.util(yd-kp)
        return up + self.β*vp

    # find optimal policy
    def policy(self, kp0, yd, v):
        lkp0 = self.spec.encode(kp0)
        dopt = lambda lkp: self.dl_bellman(lkp, yd, v)
        lkp1 = vj.optim_grad(dopt, lkp0, step=self.Δ, K=self.P)
        kp1 = self.spec.decode(lkp1)
        return kp1

    # one value iteration
    def update_interp(self, v, kp):
        # calculate updates
        v1 = self.v_bellman(kp, self.yd_grid, v)
        kp1 = self.v_policy(kp, self.yd_grid, v)

        # compute output
        err = np.max(np.abs(v1-v))
        out = {'v': v1, 'kp': kp1, 'err': err}

        # return value
        return (v1, kp1), out

    # solve for value function
    def solve_interp(self, K=400):
        # initial guess
        v0 = self.util(self.yd_grid-self.δ*self.k_grid)
        kp0 = 0.2*self.kss + 0.8*self.k_grid

        # run bellman iterations
        upd = lambda x, t: self.update_interp(*x)
        (v1, kp1), hist = lax.scan(upd, (v0, kp0), np.arange(K))

        # return full info
        _, ret = self.update_interp(v1, kp1)
        return v1, kp1, ret, hist

    # make conditional and unconditional transitions matrices
    def make_tmats(self, kp):
        tmat0p = split_range_vec(
            kp-0.5*self.k_size, kp+0.5*self.k_size, self.k_bins
        )/self.k_size
        tmat0 = tmat0p @ self.dmat
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
