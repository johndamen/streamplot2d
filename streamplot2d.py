from matplotlib import streamplot, pyplot as plt
import numpy as np
from scipy.interpolate import griddata


def streamplot2d(ax, X, Y, U, V, nx=None, ny=None, color=None, linewidth=None, scale=1, **kw):
    x = np.linspace(X.min(), X.max(), nx)
    y = np.linspace(Y.min(), Y.max(), ny)

    gX, gY = np.meshgrid(x, y)
    u = np.ma.masked_invalid(
        griddata((X.flatten(), Y.flatten()),
                 U.flatten(),
                 (gX, gY),
                 method='linear'))
    v = np.ma.masked_invalid(
        griddata(
            (X.flatten(), Y.flatten()),
            V.flatten(),
            (gX, gY),
            method='linear'))

    if isinstance(linewidth, np.ndarray):
        if linewidth.shape == U.shape:
            linewidth = scale*np.ma.masked_invalid(
                griddata(
                    (X.flatten(), Y.flatten()),
                    linewidth.flatten(),
                    (gX, gY),
                    method='linear'))
    elif linewidth == 'magnitude':
        linewidth = scale*np.absolute(u + 1j*v)**.5

    if linewidth is not None:
        kw['linewidth'] = linewidth

    if isinstance(color, np.ndarray):
        if color.shape == U.shape:
            color = np.ma.masked_invalid(
                griddata(
                    (X.flatten(), Y.flatten()),
                    color.flatten(),
                    (gX, gY),
                    method='linear'))
    elif color == 'magnitude':
        color = np.absolute(u + 1j*v)**.5

    if color is not None:
        kw['color'] = color

    return ax.streamplot(x, y, u, v, **kw)


if __name__ == '__main__':
    x = np.linspace(-3, 3, 30)
    y = np.linspace(-3, 3, 30)
    X, Y = np.meshgrid(x, y)
    X = X + .1 * Y
    Y = Y + .1 * X
    U = -1 - X ** 2 + Y
    V = 1 + X - Y ** 2

    plt.figure(figsize=(12, 10))

    ax = plt.subplot(231)
    ax.pcolor(X, Y, np.absolute(U + 1j * V), cmap='gray')
    ax.quiver(X, Y, U, V)

    ax = plt.subplot(232)
    ax.pcolor(X, Y, np.absolute(U+1j*V), cmap='gray')
    streamplot2d(ax,
                 X, Y, U, V,
                 nx=500, ny=500,
                 linewidth='magnitude', density=1, scale=.8)

    ax = plt.subplot(233)
    ax.pcolor(X, Y, np.absolute(U + 1j * V), cmap='gray')
    streamplot2d(ax,
                 X.flatten(), Y.flatten(), U.flatten(), V.flatten(),
                 nx=500, ny=500,
                 linewidth='magnitude', density=1, scale=.8)


    ax = plt.subplot(235)
    ax.pcolor(X, Y, np.absolute(U + 1j * V), cmap='gray')
    streamplot2d(ax,
                 X, Y, U, V,
                 nx=500, ny=500,
                 color='magnitude', density=1)

    ax = plt.subplot(236)
    ax.pcolor(X, Y, np.absolute(U + 1j * V), cmap='gray')
    streamplot2d(ax,
                 X.flatten(), Y.flatten(), U.flatten(), V.flatten(),
                 nx=500, ny=500,
                 color='magnitude', density=1)

    plt.show()

