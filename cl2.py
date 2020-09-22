from openTSNE import TSNE, TSNEEmbedding, affinity, initialization
from openTSNE import initialization
from openTSNE.callbacks import ErrorLogger

import plotutils as utils

import numpy as np
import matplotlib.pyplot as p



import gzip
import pickle
import pandas as pd

with gzip.open("data/macosko_2015.pkl.gz", "rb") as f:
    data = pickle.load(f)

# with open("data/retinas", "r") as f:
#     pickle.dump(data,f)
#
# with open("data/retinas", "r") as f:
#     data = pickle.load(f)

x = data["pca_50"]
y = data["CellType1"].astype(str)




# with open("data/skill_builder/chunk.pkl", "rb") as f:
#     data = pickle.load(f)
#
# x = data["problem_id"]
# y = data["skill_id"]

# print("Data set contains %d samples with %d features" % x.shape)
from openTSNE import TSNE

embedding = TSNE().fit(x)

# sns.set(rc={'figure.figsize':(11.7,8.27)})
# palette = sns.color_palette("bright",10)
# sns.scatterplot(x=embedding[:,0], y=embedding[:,1], hue=y, legend='full')
# sns.scatterplot(x=embedding[:,0], y=embedding[:,1], hue=y, legend='full')
utils.plot(embedding, y=y, colors=utils.MACOSKO_COLORS)


# plt.plot(embedding[:,0], embedding[:,1])

def plot(x, y, **kwargs):
    utils.plot(
        x,
        y,
        colors=utils.MOUSE_10X_COLORS,
        alpha=kwargs.pop("alpha", 0.1),
        draw_legend=False,
        **kwargs
    )


def rotate(degrees):
    phi = degrees * np.pi / 180
    return np.array([
        [np.cos(phi), -np.sin(phi)],
        [np.sin(phi), np.cos(phi)],
    ])


#plot(x, y)
