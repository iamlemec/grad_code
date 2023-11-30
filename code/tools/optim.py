# useful tools
def smoothmax(x, y):
    n, = x.shape

    i0 = np.argmax(y)
    im = np.maximum(0, i0 - 1)
    ip = np.minimum(n-1, i0 + 1)

    xm, x0, xp = x[im], x[i0], x[ip]
    ym, y0, yp = y[im], y[i0], y[ip]

    em, ep = x0 - xm, xp - x0
    dm, dp = y0 - ym, y0 - yp
    sm, sp = dm/em, dp/ep

    dx = 0.5*(sm*ep-sp*em)/(sm+sp)
    return x0 + dx
