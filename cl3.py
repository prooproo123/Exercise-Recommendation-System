from openTSNE import TSNE
from openTSNE.callbacks import ErrorLogger

import plotutils as utils

import numpy as np
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


import gzip
import pickle

with open("data/skill_builder/chunk.pkl", "rb") as f:
    data = pickle.load(f)

x = data["problem_id"].values
y = data["skill_id"].values

#print("Data set contains %d samples with %d features" % x.shape)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.33, random_state=42)

print("%d training samples" % x_train.shape[0])
print("%d test samples" % x_test.shape[0])


tsne = TSNE(
    perplexity=30,
    metric="euclidean",
    callbacks=ErrorLogger(),
    n_jobs=8,
    random_state=42,
)

embedding_train = tsne.fit(x_train)


utils.plot(embedding_train, y_train, colors=utils.MACOSKO_COLORS)


embedding_test = embedding_train.transform(x_test)


utils.plot(embedding_test, y_test, colors=utils.MACOSKO_COLORS)



fig, ax = plt.subplots(figsize=(8, 8))
utils.plot(embedding_train, y_train, colors=utils.MACOSKO_COLORS, alpha=0.25, ax=ax)
utils.plot(embedding_test, y_test, colors=utils.MACOSKO_COLORS, alpha=0.75, ax=ax)

