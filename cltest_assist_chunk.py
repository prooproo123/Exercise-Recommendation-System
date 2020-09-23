import pickle

import numpy as np

from clustering import plotutils as utils
import pandas as pd


def softmax(num):
    return np.exp(num) / np.sum(np.exp(num), axis=0)


def cor_weight(embedded, q):
    """
    Calculate the KCW of the exercise
    :param embedded: the embedding of exercise q
    :param q: exercise ID
    :return: the KCW of the exercise
    """
    concepts = e2c[q]

    # keymatrix num_concepts*

    corr = softmax([np.dot(embedded, key_matrix[i]) for i in concepts])
    correlation = np.zeros(Concepts)
    for j in range(len(concepts)):
        correlation[concepts[j]] = corr[j]
    return correlation


# with open('checkpoint/biology30_32batch_1epochs/kt_params', 'rb') as f:
#     params = pickle.load(f)
#
# # Knowledge Concepts Corresponding to the exercise
# # with open('data/skill_builder/chunk_exercise_concepts_mapping.pkl', 'rb') as f:
# with open('data/biology30/chunk_exercise_concepts_mapping.pkl', 'rb') as f:
#     e2c = pickle.load(f)
#
# # with open('data/skill_builder/chunk_exercises_id_converter.pkl', 'rb') as f:
# with open('data/biology30/chunk_exercises_id_converter.pkl', 'rb') as f:
#     exercises_id_converter = pickle.load(f)

with open('checkpoint/assist2009_updated_32batch_1epochs/kt_params', 'rb') as f:
    params = pickle.load(f)

with open('data/skill_builder/chunk_exercise_concepts_mapping.pkl', 'rb') as f:
    # with open('data/biology30/chunk_exercise_concepts_mapping.pkl', 'rb') as f:
    e2c = pickle.load(f)

with open('data/skill_builder/chunk_exercises_id_converter.pkl', 'rb') as f:
    # with open('data/biology30/chunk_exercises_id_converter.pkl', 'rb') as f:
    exercises_id_converter = pickle.load(f)

# with open('data/biology30/biology30.csv', 'r') as f:
#     data = pd.read_csv(f, sep='\t', index_col=None,
#                        dtype={'problem_id': int,
#                               'user_id': int,
#                               'skill_id': str,
#                               'correct': int
#                               },
#                        usecols=[
#                            'problem_id',
#                            'user_id',
#                            'skill_id',
#                            'correct'
#                        ])


with open('data/skill_builder/chunk.csv', 'r') as f:
    data = pd.read_csv(f, sep=',', index_col=None,
                       dtype={'user_id': int,
                              'problem_id': int,
                              'correct': int,
                              'skill_id': str,

                              },
                       usecols=[
                           'user_id',
                           'problem_id',
                           'correct',
                           'skill_id',

                       ])

Concepts = 9

key_matrix = params['Memory/key:0']

# with gzip.open("data/macosko_2015.pkl.gz", "rb") as f:
#     data = pickle.load(f)
#
# # with open("data/retinas", "r") as f:
# #     pickle.dump(data,f)
# #
# # with open("data/retinas", "r") as f:
# #     data = pickle.load(f)
#
# x = data["pca_50"]
# y = data["CellType1"].astype(str)

qs = data["skill_id"].values.astype(int)
q_embed_mtx = params['Embedding/q_embed:0']

with open('data/assist2009_updated/assist2009_updated_skill_mapping.txt', 'r') as f:
    lines = f.readlines(
    )

concepts_id_name = {}

for line in lines:
    concepts_id_name[int(line.split()[0])] = line.split()[1]

x = np.ndarray(shape=(data.shape[0], Concepts))
for i, q in enumerate(qs):
    x[i] = cor_weight(q_embed_mtx[q], q)

x = np.argmax(x, axis=1)
y = data["skill_id"].astype(str).values

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
utils.plot(embedding, y=y)
