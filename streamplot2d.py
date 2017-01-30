from matplotlib import streamplot, pyplot as plt
import numpy as np
from scipy.interpolate import griddata


def streamplot2d(ax, X, Y, U, V, nx=None, ny=None, color=None, linewidth=None, scale=1, **kw):
    """
    Create a streamplot from an uneven grid
    :param ax: Axes instance
    :param X: multidimensional array of x positions
    :param Y: multidimensional array of y positions of same shape as X array
    :param U: multidimensional array of u components of same shape as X array
    :param V: multidimensional array of v components of same shape as X array
    :param nx: number of points along x for interpolation
    :param ny: number of points along y for interpolation
    :param color: line color
                  also allows array of same shape as inputs
                  or "magnitude" to use vector magnitude
    :param linewidth: line width
                      also allows array of same shape as inputs
                      or "magnitude" to use vector magnitude
    :param scale: line width scaling factor when using an array or "magnitude" for linewidths
    :param kw: keyword arguments (see matplotlib.streamplot.streamplot)
    :return: see matplotlib.streamplot.streamplot
    """

    # define grid
    x = np.linspace(X.min(), X.max(), nx)
    y = np.linspace(Y.min(), Y.max(), ny)

    # interpolation coords
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

    # set linewidths
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

    # set colors
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
    
    # execute streamplot with equally spaced grid
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

