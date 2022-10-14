##
## solve/optimize tools
##

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

from . import visual as viz
from .visual import colors

def solve_univar(solver, func, state, xlim, K=100, eps=1e-8, ax=None):
    hist = [state]

    # iterate on solver update
    for i in range(K):
        state = solver(func, state)
        value = func(np.array(state))
        hist.append(state)
        if np.abs(value).max() <= eps:
            break

    # make hist into an ndarray
    hist = np.atleast_2d(np.array(hist).T)
    hist_value = func(hist)

    # plot path of solution
    if ax is None:
        _, ax = plt.subplots()
    viz.plot(func, *xlim, ax=ax, c='k')
    for x, y in zip(hist, hist_value):
        ax.scatter(x, y, zorder=10)

    return i, state, value

# hybr and lm don't work with callback
def solve_multivar(func, x0, method, axs=None):
    # store initial guess
    x0 = np.array(x0)
    f0 = func(x0)
    hist = [(x0, f0)]

    # use callback to track history
    def callback(x, f):
        hist.append((x, f))
    res = opt.root(func, x0, method=method, callback=callback)

    # reshape and get result info
    hist, hist_value = map(lambda h: np.vstack(h).T, zip(*hist))
    i, x1, f1 = res.nit, res.x, res.fun

    # this only handles 2d!
    if axs is None:
        _, axs = plt.subplots(ncols=2, figsize=(10, 3.5))

    # plot path of x values
    axs[0].plot(*hist, zorder=-1)
    axs[0].scatter(*hist)
    axs[0].scatter(*x0, color='k')
    axs[0].scatter(*x1, color=colors[1])

    # plot path of f values
    axs[1].plot(*hist_value, zorder=-1)
    axs[1].scatter(*hist_value)
    axs[1].scatter(*func(x0), color='k')
    axs[1].scatter(*func(x1), color=colors[1])

    return i, x1, f1

def optim_univar(optim, func, state, xlim, vmin=0, K=100, eps=1e-8, ax=None):
    hist = [state]

    # iterate on optim update
    for i in range(K):
        state = optim(func, state)
        value = func(np.array(state))
        hist.append(state)
        if np.abs(value).max() <= eps:
            break

    # make hist into an ndarray
    hist = np.atleast_2d(np.array(hist).T)
    hist_value = func(hist)

    if ax is None:
        _, ax = plt.subplots()
    viz.plot(func, *xlim, zero=vmin, ax=ax, c='k')
    for x, y in zip(hist, hist_value):
        ax.scatter(x, y, zorder=10)

def optim_multivar(func, x0, method, ax=None):
    # store initial guess
    x0 = np.array(x0)
    f0 = func(x0)
    hist = [(x0, f0)]

    # use callback to track history
    def callback(x):
        f = func(np.array(x))
        hist.append((x, f))
    res = opt.minimize(func, x0, method=method, callback=callback)

    # reshape and get result info
    hist, hist_value = map(lambda h: np.vstack(h).T, zip(*hist))
    i, x1, f1 = res.nit, res.x, res.fun

    # this only handles 2d!
    if ax is None:
        _, ax = plt.subplots()

    # plot path of x values
    ax.plot(*hist, zorder=-1)
    ax.scatter(*hist)
    ax.scatter(*x0, color='k')
    ax.scatter(*x1, color=colors[1])

    return i, x1, f1
