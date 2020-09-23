import pickle

import numpy as np

import plotutils as utils
import pandas as pd
from openTSNE import TSNE


def softmax(num):
    return np.exp(num) / np.sum(np.exp(num), axis=0)


def cor_weight(embedded, q):
    concepts = e2c[q]
    corr = softmax([np.dot(embedded, key_matrix[i]) for i in concepts])
    correlation = np.zeros(n_concepts)
    for j in range(len(concepts)):
        correlation[concepts[j]] = corr[j]
    return correlation


def cor_weight_all(embedded):
    corr = softmax([np.dot(embedded, key_matrix[i]) for i in range(n_concepts)])
    return corr


path_params = 'checkpoint/assist2009_updated_32batch_3epochs/kt_params'
path_e2c = 'data/skill_builder/exercise_concepts_mapping.pkl'
path_exercises_id_converter = 'data/skill_builder/concepts_id_converter.pkl'
path_skill_mapping = 'data/assist2009_updated/assist2009_updated_skill_mapping.txt'
path_data = 'data/skill_builder/skill_builder_data.csv'
data = pd.read_csv(path_data,
                   usecols=[
                       'user_id',
                       'problem_id',
                       'correct',
                       'skill_id', ])

# path_params = 'checkpoint/biology30_32batch_1epochs/kt_params'
# path_e2c = 'data/biology30/chunk_exercise_concepts_mapping.pkl'
# path_exercises_id_converter = 'data/biology30/chunk_concepts_id_converter.pkl'
# path_skill_mapping = 'data/assist2009_updated/assist2009_updated_skill_mapping.txt'
# path_data = 'data/biology30/biology30.csv'
# data = pd.read_csv(path_data, sep='\t', index_col=None,
#                    dtype={'problem_id': int,
#                           'user_id': int,
#                           'skill_id': str,
#                           'correct': int
#                           },
#                    usecols=[
#                        'problem_id',
#                        'user_id',
#                        'skill_id',
#                        'correct'
#                    ])

# path_params='checkpoint/assist2009_updated_32batch_2epochs/kt_params'
# path_e2c='data/biology30/chunk_exercise_concepts_mapping.pkl'
# path_exercises_id_converter='data/skill_builder/concepts_id_converter.pkl'
# path_skill_mapping='data/assist2009_updated/assist2009_updated_skill_mapping.txt'
# path_data='data/skill_builder/skill_builder_data.csv'
# Concepts=9


with open(path_params, 'rb') as f:
    params = pickle.load(f)

with open(path_e2c, 'rb') as f:
    e2c = pickle.load(f)

with open(path_exercises_id_converter, 'rb') as f:
    exercises_id_converter = pickle.load(f)

with open(path_skill_mapping, 'r') as f:
    lines = f.readlines()

key_matrix = params['Memory/key:0']
q_embed_mtx = params['Embedding/q_embed:0']
n_questions = len(q_embed_mtx) - 1
n_concepts = key_matrix.shape[0]

#concepts_id_name = {i:"NNNNNAAAAAA" for i in range(n_questions) }
concepts_id_name = {}

for line in lines:
    concepts_id_name[int(line.split(maxsplit=1)[0])] = line.split(maxsplit=1)[1]

c_names_set=set(concepts_id_name.keys())
c_artif=set(i for i in range(n_questions))
diff1=c_names_set.difference(c_artif)
diff2=c_artif.difference(c_names_set)

x = np.ndarray(shape=(n_questions, n_concepts))

for i in range(n_questions):
    x[i] = cor_weight_all(q_embed_mtx[i])

y = np.argmax(x, axis=1)


clusters={cl:[] for cl in np.unique(y)}

true_n_concepts=len(concepts_id_name.keys())

for i in range(true_n_concepts):
    clusters[y[i]].append(i)

for cl,list in clusters.items():
    print("Cluster: "+str(cl))
    for q in list:
        print("For q: "+str(q)+" with name "+concepts_id_name[q])

embedding = TSNE(perplexity=30).fit(x)

utils.plot(embedding, y=y, draw_centers=True)
