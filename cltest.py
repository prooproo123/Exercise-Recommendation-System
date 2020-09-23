import pickle

import numpy as np

import plotutils as utils
import pandas as pd

def softmax(num):
    return np.exp(num) / np.sum(np.exp(num), axis=0)


def cor_weight(embedded, q):
    concepts = e2c[q]
    corr = softmax([np.dot(embedded, key_matrix[i]) for i in concepts])
    correlation = np.zeros(Concepts)
    for j in range(len(concepts)):
        correlation[concepts[j]] = corr[j]
    return correlation


with open('checkpoint/biology30_32batch_1epochs/kt_params', 'rb') as f:
    params = pickle.load(f)


with open('data/biology30/chunk_exercise_concepts_mapping.pkl', 'rb') as f:
    e2c = pickle.load(f)

# with open('data/skill_builder/chunk_exercises_id_converter.pkl', 'rb') as f:
with open('data/biology30/chunk_exercises_id_converter.pkl', 'rb') as f:
    exercises_id_converter = pickle.load(f)



with open('data/biology30/biology30.csv', 'r') as f:
    data = pd.read_csv(f, sep='\t', index_col=None,
                       dtype={'problem_id': int,
                              'user_id': int,
                              'skill_id': str,
                              'correct': int
                              },
                       usecols=[
                           'problem_id',
                           'user_id',
                           'skill_id',
                           'correct'
                       ])

Concepts = 5

key_matrix = params['Memory/key:0']


qs = data["problem_id"]
q_embed_mtx = params['Embedding/q_embed:0']

x = np.ndarray(shape=(data.shape[0], Concepts))
for i, q in enumerate(qs):
    x[i] = cor_weight(q_embed_mtx[q], q)

concepts_id_name = {
    0: "Stanica",
    1: "Stanicni metabolizam",
    2: "Zivcana stanica",
    3: "Dioba stanice",
    4: "DNA",
}
y=np.argmax(x,axis=1)
#y = data["skill_id"].astype(str).values
y2=[concepts_id_name[ind] for ind in y]


from openTSNE import TSNE

embedding = TSNE().fit(x)

utils.plot(embedding, y=y2, colors=utils.BIOLOGY_COLORS)
