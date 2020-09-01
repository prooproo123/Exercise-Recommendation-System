import pickle

import numpy as np
import pandas as pd


def standard(listt):
    return str(listt).replace(" ", "").replace("[", "").replace("]", "")


def standardCon(listt, di):
    listtt = [di[ex] for ex in listt]
    return str(list(map(int, listtt))).replace(" ", "").replace("[", "").replace("]", "")

def get_mappings(chunk):

    # removing rows with nan concept value
    #df = df[df['skill_id'].notna()]

    #df['skill_id'] = df['skill_id'].astype(np.int64)

    users = chunk['user_id'].unique()
    unique_exercises = chunk['problem_id'].unique()
    unique_concepts = chunk['skill_id'].unique()

    unique_concept_onehots = np.unique(pd.get_dummies(chunk['skill_id']).values,axis=0)


    onehot_dict={k: unique_concept_onehots[k] for k in range(len(unique_concepts))}

   # outputFilePath = path_to_chunk+'/chunk_stand_ex_ind_con_ind.csv'
   # outputFile = open(outputFilePath, "w+")

    concepts_id_converter = {c: i for i, c in enumerate(unique_concepts)}
    id_concepts_converter = {i: c for i, c in enumerate(unique_concepts)}

    exercises_id_converter = {e: i for i, e in enumerate(unique_exercises)}
    id_exercises_converter = {i: e for i, e in enumerate(unique_exercises)}

    e = chunk.groupby('problem_id')['skill_id'].apply(set).apply(list).to_dict()

    exercise_concepts_mapping = {exercises_id_converter[k]: [concepts_id_converter[c] for c in v] for k, v in e.items()}

  #  concept_exercises_mapping = df.groupby('skill_id')['problem_id'].apply(set).apply(list).to_dict()
  #  user_exercises_mapping = df.groupby('user_id')['problem_id'].apply(set).apply(list).to_dict()
    # user_concepts_mapping = df.groupby('user_id')['skill_id'].apply(set).apply(list).to_dict()

   # with open(path_to_chunk+'/chunk_exercise_concepts_mapping.pkl', 'wb') as f:
     #   pickle.dump(exercise_concepts_mapping, f)

   # with open(path_to_chunk+'/chunk_concepts_id_converter.pkl', 'wb') as f:
    #    pickle.dump(concepts_id_converter, f)

   # with open(path_to_chunk+'/chunk_id_concepts_converter.pkl', 'wb') as f:
    #    pickle.dump(id_concepts_converter, f)

    #with open(path_to_chunk+'/chunk_exercises_id_converter.pkl', 'wb') as f:
    #    pickle.dump(exercises_id_converter, f)

    #with open(path_to_chunk+'/chunk_id_exercises_converter.pkl', 'wb') as f:
      #  pickle.dump(id_exercises_converter, f)

    temp = [[] for i in range(len(users))]
    # t1=df.groupby('user_id')['problem_id','correct','skill_id']

    return exercise_concepts_mapping,exercises_id_converter
