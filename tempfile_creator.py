import pickle

import numpy as np
import pandas as pd


def standard(listt):
    return str(listt).replace(" ", "").replace("[", "").replace("]", "")


def standardCon(listt, di):
    listtt = [di[ex] for ex in listt]
    return str(list(map(int, listtt))).replace(" ", "").replace("[", "").replace("]", "")

def create_files(path_to_chunk):
    df = pd.read_csv(path_to_chunk)

    # removing rows with nan concept value
    df = df[df['skill_id'].notna()]

    df['skill_id'] = df['skill_id'].astype(np.int64)

    users = df['user_id'].unique()
    unique_exercises = df['problem_id'].unique()
    unique_concepts = df['skill_id'].unique()

    unique_concept_onehots = np.unique(pd.get_dummies(df['skill_id']).values,axis=0)


    onehot_dict={k: unique_concept_onehots[k] for k in range(len(unique_concepts))}

    outputFilePath = path_to_chunk+'/chunk_stand_ex_ind_con_ind.csv'
    outputFile = open(outputFilePath, "w+")

    concepts_id_converter = {c: i for i, c in enumerate(unique_concepts)}
    id_concepts_converter = {i: c for i, c in enumerate(unique_concepts)}

    exercises_id_converter = {e: i for i, e in enumerate(unique_exercises)}
    id_exercises_converter = {i: e for i, e in enumerate(unique_exercises)}

    e = df.groupby('problem_id')['skill_id'].apply(set).apply(list).to_dict()

    exercise_concepts_mapping = {exercises_id_converter[k]: [concepts_id_converter[c] for c in v] for k, v in e.items()}

    concept_exercises_mapping = df.groupby('skill_id')['problem_id'].apply(set).apply(list).to_dict()
    user_exercises_mapping = df.groupby('user_id')['problem_id'].apply(set).apply(list).to_dict()
    user_concepts_mapping = df.groupby('user_id')['skill_id'].apply(set).apply(list).to_dict()

    with open(path_to_chunk+'/chunk_exercise_concepts_mapping.pkl', 'wb') as f:
        pickle.dump(exercise_concepts_mapping, f)

    with open(path_to_chunk+'/chunk_concepts_id_converter.pkl', 'wb') as f:
        pickle.dump(concepts_id_converter, f)

    with open(path_to_chunk+'/chunk_id_concepts_converter.pkl', 'wb') as f:
        pickle.dump(id_concepts_converter, f)

    with open(path_to_chunk+'/chunk_exercises_id_converter.pkl', 'wb') as f:
        pickle.dump(exercises_id_converter, f)

    with open(path_to_chunk+'/chunk_id_exercises_converter.pkl', 'wb') as f:
        pickle.dump(id_exercises_converter, f)

    temp = [[] for i in range(len(users))]
    # t1=df.groupby('user_id')['problem_id','correct','skill_id']

    max_exercises = 0
    max_concepts = 0
    l = []
    for i, student in enumerate(users):
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

        exercises = [exercises_id_converter[e] for e in exercises]
        concepts = [concepts_id_converter[c] for c in concepts]

        if len(exercises) > max_exercises:
            max_exercises = len(exercises)

        if len(concepts) > max_concepts:
            max_concepts = len(concepts)

        if len(exercises) != len(concepts):
            print("razlika je" + str(len(exercises) - len(concepts)))

        temp[i].append(exercises)

        n_questions = len(unique_exercises)

        temp[i].append([answers[j] * n_questions + exercises[j] for j in range(len(exercises))])

        temp[i].append([[c] for c in concepts])
        concept_onehots = [onehot_dict[c] for c in concepts]
        temp[i].append(concept_onehots)
        #concepts 9 or 5
        # if len(exercises) < 1900:
        l.append(len(exercises))
        # outputFile.write(str(len(exercises)) + "\n")
        # outputFile.write(standardCon(exercises, exercises_id_converter) + "\n")
        # outputFile.write(standard(answers) + "\n")
        # outputFile.write(standardCon(concepts, concepts_id_converter) + "\n")

    l2 = sorted(l, reverse=True)

    with open(path_to_chunk+'/temp.pkl', 'wb') as f:
        pickle.dump(temp, f)

    outputFile.close()



#main

create_files('/Biologija')