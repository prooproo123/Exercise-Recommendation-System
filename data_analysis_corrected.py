import pandas as pd

# sb09 = pd.read_csv('data/skill_builder/skill_builder_data.csv')
sb09_updated = pd.read_csv('data/skill_builder/skill_builder_data_corrected_collapsed.csv',
                           dtype={'order_id': int, 'assignment_id': int, 'user_id': int, 'assistment_id': int,
                                  'problem_id': int,
                                  'original': int, 'correct': int, 'attempt_count': int, 'ms_first_response': int,
                                  'tutor_mode': 'string', 'answer_type': 'string', 'sequence_id': int,
                                  'student_class_id': int,
                                  'position': int, 'type': 'string', 'base_sequence_id': int, 'skill_id': 'string',
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
sb09_updated = sb09_updated[sb09_updated['skill_id'] != 'NA']

users_updated = sb09_updated['user_id'].unique()
exercises_updated = sb09_updated['problem_id'].unique()
concepts_updated = sb09_updated['skill_id'].unique()

simple_concepts_updated = set(int(s) for c in concepts_updated for s in c.split('_'))

# exercise_concepts_mapping=sb09_updated.groupby('problem_id')['skill_id'].apply(set).to_dict()
# concept_exercises_mapping=sb09_updated.groupby('skill_id')['problem_id'].apply(set).to_dict()

# sb09_updated[sb09_updated[]]

print('Skill builder 2009 UPDATED dataset:')
print('Number of students: ' + str(len(users_updated)))
print('Number of exercises: ' + str(len(exercises_updated)))
print('Number of composite concepts: ' + str(len(concepts_updated)))
print('Number of simple concepts: ' + str(len(simple_concepts_updated)))
print()

sb15 = pd.read_csv('data/skill_builder/sb_100.csv')
sb09_updated = sb15[sb15['sequence_id'] != 'NA']
users15 = sb15['user_id'].unique()
logs15 = sb15['log_id'].unique()
sequences15 = sb15['sequence_id'].unique()

# sb15[sb15[]]

print('Skill builder main 100 problems 2015 dataset:')
print('Number of students: ' + str(len(users15)))
print('Number of logs: ' + str(len(logs15)))
print('Number of sequences: ' + str(len(sequences15)))