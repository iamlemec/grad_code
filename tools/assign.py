# assignment tools

import jax
import jax.numpy as np
from functools import partial

# vector tools
# vector_interp = jax.vmap(np.interp, in_axes=(0, None, None))

# defined functions
def util(c, ϵ=1e-6):
    c1 = np.maximum(ϵ, c)
    return np.log(c1)
def prod(k, z, α):
    return z*k**α

# find steady state
def calc_kss(par):
    β, δ, z, α = par['β'], par['δ'], par['z'], par['α']
    rhs = (1-β)/β + δ
    k_ss = (α*z/rhs)**(1/(1-α))
    return k_ss

# grid and steady state
def make_grid(k_ss, f_lo, f_hi, N):
    k_min, k_max = f_lo*k_ss, f_hi*k_ss
    k_grid = np.linspace(k_min, k_max, N)
    return k_grid

def bellman(kp, yp, vp, β):
    kp1 = np.maximum(0.0, kp)
    return util(yp-kp1) + β*vp

# interpolated bellman calculation
def bellman_interp(kp, yp, k, v, β):
    vp = np.interp(kp, k, v)
    return bellman(kp, yp, vp, β)
d_bellman_interp = jax.grad(bellman_interp, argnums=0)
v_bellman_interp = jax.vmap(bellman_interp, in_axes=(0, 0, None, None, None))
vd_bellman_interp = jax.vmap(d_bellman_interp, in_axes=(0, 0, None, None, None))

# update using intpolation
def update_interp(par, grid, kp, vp, Δ=0.01):
    β = par['β']
    k = grid['k']
    yp = grid['yp']

    # update policy
    dvdk = vd_bellman_interp(kp, yp, k, vp, β)
    k1 = np.clamp(kp + Δ*dvdk, k_lo, k_hi)

    # update value
    vx = bellman(kp, yp, vp, β)
    v1 = Δ*vx + (1-Δ)*vp

    return k1, v1

# execute interpolation solve
def solve_interp(par, k_grid, Δ=0.01, K=500):
    α, z, δ = par['α'], par['z'], par['δ']

    # precompute grids
    y_grid = prod(k_grid, z, α)
    yp_grid = y_grid + (1-δ)*k_grid
    grid = {
        'k': k_grid,
        'yp': yp_grid,
    }

    # apply to value update
    value1 = partial(update_interp, par, grid)

    # initial guesses
    k, v = k_grid, util(k_grid)

    # iterate on update
    for i in range(K):
        kp, vp = value1(k, v, Δ=Δ)

        err_k = np.max(np.abs(kp-k))
        err_v = np.max(np.abs(vp-v))
        err = np.maximum(err_k, err_v)
        if i == 0 or i == K - 1 or i % 100 == 0:
            print(i, err_k, err_v)

        k, v = kp, vp

    return kp, vp
