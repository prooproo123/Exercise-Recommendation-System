# import palette as palette
# import sns as sns
# import numpy as np
# from sklearn.datasets import load_digits
# from scipy.spatial.distance import pdist
# from sklearn.manifold.t_sne import _joint_probabilities
# from scipy import linalg
# from sklearn.metrics import pairwise_distances
# from scipy.spatial.distance import squareform
# from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import datasets

import plotutils as utils




iris = datasets.load_iris()
x, y = iris["data"], iris["target"]

from openTSNE import TSNE

embedding = TSNE().fit(x)

# sns.set(rc={'figure.figsize':(11.7,8.27)})
# palette = sns.color_palette("bright",10)
# sns.scatterplot(x=embedding[:,0], y=embedding[:,1], hue=y, legend='full')
# sns.scatterplot(x=embedding[:,0], y=embedding[:,1], hue=y, legend='full')
utils.plot(embedding,y=y, colors=utils.MACOSKO_COLORS)
#plt.plot(embedding[:,0], embedding[:,1])