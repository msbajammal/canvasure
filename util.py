
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def pad_zeros_around_image(image, pad_width=10):
    orig_height, orig_width = image.shape
    padded_image = np.zeros((orig_height+2*pad_width, orig_width+2*pad_width))
    padded_image[pad_width:pad_width+orig_height, pad_width:pad_width+orig_width] = image
    return padded_image

from math import floor, isinf, isnan

def truncate(f, n):
    if isinf(f) or isnan(f):
        return f
    else:
        return floor(f * 10 ** n) / 10 ** n

def imshow_hough(hspace, angles, dists):
    fig = plt.figure()
    plt.imshow(
    hspace,
    extent=(np.rad2deg(angles[-1]), np.rad2deg(angles[0]), dists[-1], dists[0]))
    fig.canvas.manager.window.raise_()
    return fig

def popup_imshow_surface(gray_image, title='', xlabel='', ylabel='', zlabel=''):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = np.arange(0, gray_image.shape[1])
    y = np.arange(0, gray_image.shape[0])
    X, Y = np.meshgrid(x, y)

    def fun(x, y):
        return gray_image[y, x]

    zs = np.array([fun(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    ax.plot_surface(X, Y, Z)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()
    fig.canvas.manager.window.raise_()

    return fig

def popup_imshow(img, title='', cmap=None):
    fig = plt.figure()

    plt.imshow(img, cmap)
    plt.axis('off')
    plt.title(title)
    plt.tight_layout()

    fig.canvas.manager.window.raise_()

    return fig
