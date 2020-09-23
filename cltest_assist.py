import pickle

import numpy as np

from clustering import plotutils as utils
import pandas as pd
from openTSNE import TSNE


def softmax(num):
    return np.exp(num) / np.sum(np.exp(num), axis=0)


def cor_weight(embedded):
    """
    Calculate the KCW of the exercise
    :param embedded: the embedding of exercise q
    :param q: exercise ID
    :return: the KCW of the exercise
    """
    #concepts = e2c[q]

    # keymatrix num_concepts*

    corr = softmax([np.dot(embedded, key_matrix[i]) for i in range(Concepts)])
    # correlation = np.zeros(Concepts)
    # for j in range(Concepts):
    #     correlation[j] = corr[j]
    return corr


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

with open('checkpoint/assist2009_updated_32batch_2epochs/kt_params', 'rb') as f:
    params = pickle.load(f)

# with open('data/skill_builder/exercise_concepts_mapping.pkl', 'rb') as f:
#     # with open('data/biology30/chunk_exercise_concepts_mapping.pkl', 'rb') as f:
#     e2c = pickle.load(f)

with open('data/skill_builder/concepts_id_converter.pkl', 'rb') as f:
    # with open('data/biology30/chunk_exercises_id_converter.pkl', 'rb') as f:
    exercises_id_converter = pickle.load(f)

with open('data/assist2009_updated/assist2009_updated_skill_mapping.txt', 'r') as f:
    lines = f.readlines()

concepts_id_name = {}

for line in lines:
    concepts_id_name[int(line.split()[0])] = line.split()[1]

data = pd.read_csv('data/skill_builder/skill_builder_data.csv',
                   dtype={'order_id': int, 'assignment_id': int, 'user_id': int, 'assistment_id': int,
                          'problem_id': int,
                          'original': int, 'correct': int, 'attempt_count': int, 'ms_first_response': int,
                          'tutor_mode': 'string', 'answer_type': 'string', 'sequence_id': int,
                          'student_class_id': int,
                          'position': int, 'type': 'string', 'base_sequence_id': int, 'skill_id': float,
                          'skill_name': 'string',
                          'teacher_id': int, 'school_id': int, 'hint_count': int, 'hint_total': int,
                          'overlap_time': int,
                          'template_id': int, 'answer_id': int, 'answer_text': 'string',
                          'first_action': int,
                          'bottom_hint': int, 'opportunity': int, 'opportunity_original': int
                          },
                   usecols=[
                       'user_id',
                       'problem_id',
                       'correct',
                       'skill_id',

                   ])

#candidate_exercises = [exercises_id_converter[e] for e in cands]

Concepts = 20

key_matrix = params['Memory/key:0']


q_embed_mtx = params['Embedding/q_embed:0']

#for now just unique-testing
#qs = data["skill_id"].unique().astype(int)
x = np.ndarray(shape=(110, Concepts))
#qs2=[exercises_id_converter[e] for e in qs]

for i in range(110):
    x[i] = cor_weight(q_embed_mtx[i])

y = np.argmax(x, axis=1)
#y = data["skill_id"].astype(str).values

embedding = TSNE().fit(x)

utils.plot(embedding, y=y,draw_centers=True)
