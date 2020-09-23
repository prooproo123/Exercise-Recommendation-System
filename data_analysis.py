import pickle

import pandas as pd
import numpy as np
from statistics import mean, median
# DEBUGGING PKL FILES

# zadaci kandidati za preporuku
with open('data/skill_builder/kt_params.pkl', 'rb') as f:
    a = pickle.load(f)


with open('data/skill_builder/old_cand_ex.pkl', 'rb') as f:
    x = pickle.load(f)

with open('/home/zvonimir/Exercise-Recommendation-System/checkpoint/a_32batch_1epochs/kt_params', 'rb') as f:
    b = pickle.load(f)

with open('checkpoint/assist2009_updated_32batch_3epochs/kt_params', 'rb') as f:
    assist2009_updated_32batch_3epochs = pickle.load(f)

with open('checkpoint/STATICS_10batch_3epochs/kt_params', 'rb') as f:
    STATICS_10batch_3epochs = pickle.load(f)

# dict poveznica koncepti-zadaci
with open('data/skill_builder/old_cand_ex.pkl', 'rb') as f:
    arms = pickle.load(f)
# utrenirani parametri DKVMN-CA modela
with open('data/skill_builder/candidates_pickled.pkl', 'rb') as f:
    cands = pickle.load(f)

with open('data/skill_builder/old_e2c.pkl', 'rb') as f:
    e2c_old = pickle.load(f)

with open('data/skill_builder/old_kt_params.pkl', 'rb') as f:
    parametri_kt_modela = pickle.load(f)

debug = 0

# with open('data/skill_builder/stan_con.csv', 'rb') as f:
#     stan_con = pickle.load(f)
#
#
# with open('data/skill_builder/stand_exind_nocon.csv', 'rb') as f:
#     stand_exind_nocon = pickle.load(f)
#
# with open('data/skill_builder/skill_builder_data.csv', 'rb') as f:
#     parametri_kt_modela = pickle.load(f)

with open('data/skill_builder/skill_builder_pickle.pkl', 'rb') as f:
    skill_builder_pickled_data = pickle.load(f)

# sb09 = pd.read_csv('data/skill_builder/skill_builder_data.csv')
sb09 = pd.read_csv('data/skill_builder/skill_builder_data.csv',
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
                   usecols=['order_id', 'assignment_id', 'user_id', 'assistment_id', 'problem_id',
                            'original',
                            'correct',
                            'attempt_count', 'ms_first_response', 'tutor_mode', 'answer_type',
                            'sequence_id',
                            'student_class_id', 'position', 'type', 'base_sequence_id', 'skill_id',
                            'skill_name',
                            'teacher_id', 'school_id', 'hint_count', 'hint_total', 'overlap_time',
                            'template_id',
                            'first_action', 'opportunity'])

# removing rows with nan concept value
sb09 = sb09[sb09['skill_id'].notna()]

sb09['skill_id']=sb09['skill_id'].astype(np.int64)

users = sb09['user_id'].unique()
exercises = sb09['problem_id'].unique()
concepts = sb09['skill_id'].unique()

concepts_id_converter={c:i for i,c in enumerate(concepts)}
id_concepts_converter={i:c for i,c in enumerate(concepts)}

exercises_concepts_converter={e:i for i,e in enumerate(exercises)}
id_exercises_converter={i:e for i,e in enumerate(exercises)}

exercise_concepts_mapping=sb09.groupby('problem_id')['skill_id'].apply(set).apply(list).to_dict()
concept_exercises_mapping=sb09.groupby('skill_id')['problem_id'].apply(set).apply(list).to_dict()
user_exercises_mapping=sb09.groupby('user_id')['problem_id'].apply(set).apply(list).to_dict()
user_concepts_mapping=sb09.groupby('user_id')['skill_id'].apply(set).apply(list).to_dict()

with open('data/skill_builder/e2c.pkl', 'wb') as f:
    pickle.dump(exercise_concepts_mapping,f)

print('Skill builder 2009 dataset:')
print('Number of students: ' + str(len(users)))
print('Number of exercises: ' + str(len(exercises)))
print('Number of concepts: ' + str(len(concepts)))
print()

print('Maximum exercises per student: '+str(max([len(vals) for vals in user_exercises_mapping.values()])))
print('Maximum concepts per student: '+str(max([len(vals) for vals in user_concepts_mapping.values()])))
print('Maximum exercises per concept: '+str(max([len(vals) for vals in concept_exercises_mapping.values()])))
print('Maximum concepts per exercise: '+str(max([len(vals) for vals in exercise_concepts_mapping.values()])))
print()

print('Median exercises per student: '+str(median([len(vals) for vals in user_exercises_mapping.values()])))
print('Median concepts per student: '+str(median([len(vals) for vals in user_concepts_mapping.values()])))
print('Median exercises per concept: '+str(median([len(vals) for vals in concept_exercises_mapping.values()])))
print('Median concepts per exercise: '+str(median([len(vals) for vals in exercise_concepts_mapping.values()])))
print()
