##
## solve/optimize tools
##

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

from . import visual as viz

def solve_univar(solver, func, state, xlim, K=100, eps=1e-8, ax=None):
    hist = [state]
    conv = False

    for i in range(K):
        state = solver(func, state)
        value = func(np.array(state))
        hist.append(state)
        if np.abs(value).max() <= eps:
            conv = True
            break

    hist = np.atleast_2d(np.array(hist).T)
    hist_value = func(hist)

    if ax is None:
        _, ax = plt.subplots()
    viz.plot(func, *xlim, ax=ax, c='k')
    for x, y in zip(hist, hist_value):
        ax.scatter(x, y, zorder=10)

    return i, state, value

def solve_multivar(func, x0, method, ax=None):
    hist = []
    def callback(x, f):
        hist.append((x, f))
    res = opt.root(func, x0, method=method, callback=callback)

    hist, hist_value = map(lambda h: np.vstack(h).T, zip(*hist))
    K, T = hist.shape
    i, x1, f1 = res.nit, res.x, res.fun

    if ax is None:
        _, axs = plt.subplots(ncols=K, figsize=(5*K, 3.5))
    for ax, xi, xs, xk in zip(axs, x0, x1, hist):
        ax.plot(np.hstack([xi, xk]))
        ax.hlines(xs, *ax.get_xlim(), color='k', linewidth=1, linestyle='--')

    return T, x1, f1
