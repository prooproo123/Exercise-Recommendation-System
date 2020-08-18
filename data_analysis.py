import pickle

import pandas as pd

# DEBUGGING PKL FILES

# zadaci kandidati za preporuku
with open('data/skill_builder/new_kt_params.pkl', 'rb') as f:
    a = pickle.load(f)
# dict poveznica koncepti-zadaci
with open('data/skill_builder/old_cand_ex.pkl', 'rb') as f:
    arms = pickle.load(f)
# utrenirani parametri DKVMN-CA modela
with open('data/skill_builder/candidates_pickled.pkl', 'rb') as f:
    cands = pickle.load(f)

with open('data/skill_builder/e2c_old.pkl', 'rb') as f:
    e2c_old = pickle.load(f)

with open('data/skill_builder/kt_params.pkl', 'rb') as f:
    parametri_kt_modela = pickle.load(f)

b = 0

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

#removing rows with nan concept value
sb09 = sb09[sb09['skill_id'].notna()]

users = sb09['user_id'].unique()
exercises = sb09['problem_id'].unique()
concepts = sb09['skill_id'].unique()

# sb09[sb09[]]

print('Skill builder 2009 dataset:')
print('Number of students: ' + str(len(users)))
print('Number of exercises: ' + str(len(exercises)))
print('Number of concepts: ' + str(len(concepts)))
print()


# sb09 = pd.read_csv('data/skill_builder/skill_builder_data.csv')
sb09_updated = pd.read_csv('data/skill_builder/skill_builder_data_corrected_collapsed.csv',
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

#removing rows with nan concept value
#sb09_updated = sb09_updated[sb09_updated['skill_id'].notna()]

users = sb09_updated['user_id'].unique()
exercises = sb09_updated['problem_id'].unique()
concepts = sb09_updated['skill_id'].unique()

# sb09_updated[sb09_updated[]]

print('Skill builder 2009 UPDATED dataset:')
print('Number of students: ' + str(len(users)))
print('Number of exercises: ' + str(len(exercises)))
print('Number of concepts: ' + str(len(concepts)))
print()




sb15 = pd.read_csv('data/skill_builder/sb_100.csv')
users = sb15['user_id'].unique()
logs = sb15['log_id'].unique()
sequences = sb15['sequence_id'].unique()

# sb15[sb15[]]

print('Skill builder main 100 problems 2015 dataset:')
print('Number of students: ' + str(len(users)))
print('Number of logs: ' + str(len(logs)))
print('Number of sequences: ' + str(len(sequences)))
