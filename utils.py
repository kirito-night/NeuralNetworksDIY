import numpy as np

from mltools import *
def proj_biais(datax):
    """ Ajoute une colonne de 1 a datax
    :param datax: matrice des donnees
    :return: matrice des donnees + colonne de 1
    """
    return np.hstack((datax,np.ones((datax.shape[0],1))))



from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def plot_frontiere_3d(ax, data, f, step=20):
    """Trace un graphe de la frontiere de decision de f en 3D
    :param data: donnees
    :param f: fonction de decision
    :param step: pas de la grille
    :return:
    """
    grid, x, y = make_grid(data=data, step=step)
    z = f(grid).reshape(x.shape)
    ax.plot_surface(x, y, z)


def plot_data_3d(ax, data, labels=None):
    """
    Affiche des donnees 3D
    :param data: matrice des donnees 2d
    :param labels: vecteur des labels (discrets)
    :return:
    """
    if labels is not None:
        labels = labels.reshape(-1)
    cols, marks = ["red", "green", "blue", "orange", "black", "cyan"], [
        ".",
        "+",
        "*",
        "o",
        "x",
        "^",
    ]

    if labels is None:
        ax.scatter(data[:, 0], data[:, 1], 0, marker="x")
        return
    # add label to each point as 3rd dimension
    for i, l in enumerate(sorted(list(set(labels.flatten())))):
        ax.scatter(
            data[labels == l, 0],
            data[labels == l, 1],
            labels[labels == l],
            c=cols[i],
            marker=marks[i],
        )


def make_minibatch(batch_size, x_data, y_data):
    n = x_data.shape[0]
    n_batch = max(n // batch_size, 1)
    index = np.random.permutation(n)
    return np.array_split(index, n_batch)


def load_usps(fn):
    with open(fn,"r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp=np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)

def get_usps(l,datax,datay):
    if type(l)!=list:
        resx = datax[datay==l,:]
        resy = datay[datay==l]
        return resx,resy
    tmp =   list(zip(*[get_usps(i,datax,datay) for i in l]))
    tmpx,tmpy = np.vstack(tmp[0]),np.hstack(tmp[1])
    return tmpx,tmpy

def show_usps(data):
    plt.imshow(data.reshape((16,16)),interpolation="nearest",cmap="gray")

def onehot(Y): 
    n = Y.shape[0]
    k = np.unique(Y).shape[0]
    res = np.zeros((n,k))
    res[np.arange(n),Y] = 1
    return res