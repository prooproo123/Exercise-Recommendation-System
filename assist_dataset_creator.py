
import pandas as pd
import numpy as np
import pickle

assistments_data_path = 'data/skill_builder/skill_builder_data.csv'


def standard(listt):
    return str(listt).replace(" ", "").replace("[", "").replace("]", "")


def standardCon(listt, di):
    listtt = [di[ex] for ex in listt]
    return str(list(map(int, listtt))).replace(" ", "").replace("[", "").replace("]", "")


df = pd.read_csv('data/skill_builder/skill_builder_data.csv',
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
df = df[df['skill_id'].notna()]

df['skill_id']=df['skill_id'].astype(np.int64)

users = df['user_id'].unique()
exercises = df['problem_id'].unique()
concepts = df['skill_id'].unique()

outputFilePath = 'data/skill_builder/stand_ex_ind_con_ind.csv'
outputFile = open(outputFilePath, "w+")


concepts_id_converter={c:i for i,c in enumerate(concepts)}
id_concepts_converter={i:c for i,c in enumerate(concepts)}

exercises_id_converter={e:i for i,e in enumerate(exercises)}
id_exercises_converter={i:e for i,e in enumerate(exercises)}

e=df.groupby('problem_id')['skill_id'].apply(set).apply(list).to_dict()

exercise_concepts_mapping={exercises_id_converter[k]:[concepts_id_converter[c] for c in v] for k,v in e.items()}

concept_exercises_mapping=df.groupby('skill_id')['problem_id'].apply(set).apply(list).to_dict()
user_exercises_mapping=df.groupby('user_id')['problem_id'].apply(set).apply(list).to_dict()
user_concepts_mapping=df.groupby('user_id')['skill_id'].apply(set).apply(list).to_dict()

with open('data/skill_builder/exercise_concepts_mapping.pkl', 'wb') as f:
    pickle.dump(exercise_concepts_mapping,f)

with open('data/skill_builder/concepts_id_converter.pkl', 'wb') as f:
    pickle.dump(concepts_id_converter,f)

with open('data/skill_builder/id_concepts_converter.pkl', 'wb') as f:
    pickle.dump(id_concepts_converter,f)

with open('data/skill_builder/exercises_id_converter.pkl', 'wb') as f:
    pickle.dump(exercises_id_converter,f)

with open('data/skill_builder/id_exercises_converter.pkl', 'wb') as f:
    pickle.dump(id_exercises_converter,f)


max_exercises = 0
max_concepts = 0
l = []
for student in users:
    condition = df['user_id'] == student
    studentData = df[condition]
    exercises = studentData['problem_id'].values
    answers = studentData['correct'].values
    concepts = studentData['skill_id'].values
    con = np.isnan(concepts)
    concepts = concepts[~con]
    exercises = exercises[~con]
    # time = studentData['skill_id'].tolist()
    # difficulty = studentData['skill_id'].tolist()
    # gate = studentData['skill_id'].tolist()
    exercises = exercises.tolist()
    answers = answers.tolist()
    concepts = concepts.tolist()

    if len(exercises) > max_exercises:
        max_exercises = len(exercises)

    if len(concepts) > max_concepts:
        max_concepts = len(concepts)

    if len(exercises) != len(concepts):
        print("razlika je" + str(len(exercises) - len(concepts)))

    # if len(exercises) < 1900:
    l.append(len(exercises))
    outputFile.write(str(len(exercises)) + "\n")
    outputFile.write(standardCon(exercises, exercises_id_converter) + "\n")
    outputFile.write(standard(answers) + "\n")
    #outputFile.write(standardCon(concepts,concepts_id_converter) + "\n")

l2 = sorted(l, reverse=True)
outputFile.close()

