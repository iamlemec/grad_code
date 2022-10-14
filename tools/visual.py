##
## visualization tools
##

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

colors = mpl.rcParams['axes.prop_cycle'].by_key()['color']

# plot a function over a given range
# optionally include dashed zero line
def plot(f, x0, x1, N=100, ax=None, zero=0, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    x = np.linspace(x0, x1, N)
    y = f(x)
    ax.plot(x, y, **kwargs)
    if zero is not None:
        ax.hlines(zero, *ax.get_xlim(), color='k', linestyle='--', linewidth=1, zorder=-1)
