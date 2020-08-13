# coding: utf-8

from __future__ import division

import pickle
import numpy as np

assistments_pickled = '/home/zvonimir/Exercise-Recommendation-System/data/skill_builder_pickle.pkl'

with open(assistments_pickled, 'rb') as f:
    df = pickle.load(f)

exercises = df['problem_id'].unique()
concepts = df['skill_id'].unique()

concepts_exercises_dict = dict.fromkeys(concepts)
exercises_concepts_dict = dict.fromkeys(exercises)

for concept in concepts:
    condition2 = df['skill_id'] == concept
    concepts_exercises_dict[concept] = set(df[condition2]['problem_id'])

for exercise in exercises:
    condition3 = df['problem_id'] == exercise
    exercises_concepts_dict[exercise] = set(df[condition3]['skill_id'])

# users 64525 and 70363
# TODO synthesize this and pickle
student_traces = [[(51424, 1), (51435, 1)], [(51444, 0), (51395, 1), (51481, 0)]]

all_candidate_exercises = []
for trace in student_traces:
    candidate_exercises = set()
    candidate_concepts = set()
    for exercise, answer in trace:
        candidate_concepts.update(exercises_concepts_dict[exercise])
    candidate_concepts = list(candidate_concepts)
    for c in candidate_concepts:
        candidate_exercises.update(concepts_exercises_dict[c])
    all_candidate_exercises.append(list(candidate_exercises))

with open('candidates_pickled.pkl', 'wb') as f:
    pickle.dump(all_candidate_exercises, f)

new_list=[]
with open('candidates_pickled.pkl', 'rb') as f:
    new_list = pickle.load(f)

a = 0
