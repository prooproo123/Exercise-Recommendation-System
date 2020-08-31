import pickle

import pandas as pd
import numpy as np
from statistics import mean, median
# DEBUGGING PKL FILES

cands = []
stu = []
stu_list = []

# sb09 = pd.read_csv('data/skill_builder/skill_builder_data.csv')
sb09 = pd.read_csv('/content/gdrive/My Drive/data/skill_builder_data.csv',
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


n = 40000  #chunk row size
list_df = [sb09[i:i+n] for i in range(0,sb09.shape[0],n)]
chunk=list_df[3]
# with open('data/skill_builder/chunk.pkl', 'wb') as f:
#     pickle.dump(d,f)
chunk = chunk[chunk['skill_id'].notna()]

chunk['skill_id'] = chunk['skill_id'].astype(np.int64)
chunk.to_csv('data/skill_builder/chunk.csv', columns=['user_id', 'problem_id', 'correct', 'skill_id'])

#zasada identicno kao sto je bilo hardkodirano u originalnom fajlu
#proba u df pa uzimanje po stupcima kandidata za cands
df = pd.read_csv('data/skill_builder/chunk.csv')
#print('chunk', df.head())
for index, row in df.iterrows():
        if index>3:
          break
        cands.append(row['problem_id'])

for index, row in df.iterrows():
        if index>3:
          break
        if index%2:
          continue
        stu.append((row['problem_id'],row['correct']))

stu_list.append(stu)

print('cands', cands)
print('stu', stu)  
print('stu list', stu_list)   

#print('chunk problem id', chunk['problem_id'])

for j in range(len(list_df)-2):
    print('For chunk '+str(j)+': \n')

    sb09=list_df[j]

    # removing rows with nan concept value
    sb09 = sb09[sb09['skill_id'].notna()]

    sb09['skill_id']=sb09['skill_id'].astype(np.int64)

    users = sb09['user_id'].unique()
    exercises = sb09['problem_id'].unique()
    concepts = sb09['skill_id'].unique()

    exercise_concepts_mapping=sb09.groupby('problem_id')['skill_id'].apply(set).apply(list).to_dict()
    concept_exercises_mapping=sb09.groupby('skill_id')['problem_id'].apply(set).apply(list).to_dict()
    user_exercises_mapping=sb09.groupby('user_id')['problem_id'].apply(set).apply(list).to_dict()
    user_concepts_mapping=sb09.groupby('user_id')['skill_id'].apply(set).apply(list).to_dict()

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

def exer_conc():
  return len(exercises), len(concepts)

def cand_stu():
  return cands, stu_list

