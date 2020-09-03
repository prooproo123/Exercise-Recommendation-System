import pickle

import numpy as np
import pandas as pd

class ChunkMappings:
    def __init__(self,con_id_conv=None,id_con_conv=None,exer_id_conv=None,id_exer_conv=None,
             exer_con_map=None,con_exer_map=None,user_exer_map=None,user_con_map=None):
        self.con_id_conv=con_id_conv
        self.id_con_conv=id_con_conv
        self.exer_id_conv=exer_id_conv
        self.id_exer_conv=id_exer_conv
        self.exer_con_map=exer_con_map
        self.con_exer_map=con_exer_map
        self.user_exer_map=user_exer_map
        self.user_con_map=user_con_map

    def get_con_id_conv(self):
        return self.con_id_conv

    def get_id_con_conv(self):
        return self.id_con_conv

    def get_exer_id_conv(self):
        return self.exer_id_conv

    def get_id_exer_conv(self):
        return self.id_exer_conv

    def get_exer_con_map(self):
        return self.exer_con_map

    def get_con_exer_map(self):
        return self.con_exer_map

    def get_user_exer_map(self):
        return self.user_exer_map

    def get_user_con_map(self):
        return self.user_con_map

def standard(listt):
    return str(listt).replace(" ", "").replace("[", "").replace("]", "")


def standardCon(listt, di):
    listtt = [di[ex] for ex in listt]
    return str(list(map(int, listtt))).replace(" ", "").replace("[", "").replace("]", "")

def get_mappings(chunk):


    users = chunk['user_id'].unique()
    unique_exercises = chunk['problem_id'].unique()
    unique_concepts = chunk['skill_id'].unique()

    unique_concept_onehots = np.unique(pd.get_dummies(chunk['skill_id']).values,axis=0)


    onehot_dict={k: unique_concept_onehots[k] for k in range(len(unique_concepts))}

    #key-pravi id koncepta, value- prilagodeni id koncepta
    concepts_id_converter = {c: i for i, c in enumerate(unique_concepts)}

    id_concepts_converter = {i: c for i, c in enumerate(unique_concepts)}

    exercises_id_converter = {e: i for i, e in enumerate(unique_exercises)}
    id_exercises_converter = {i: e for i, e in enumerate(unique_exercises)}

    e = chunk.groupby('problem_id')['skill_id'].apply(set).apply(list).to_dict()

    exercise_concepts_mapping = {exercises_id_converter[k]: [concepts_id_converter[c] for c in v] for k, v in e.items()}

  #  concept_exercises_mapping = df.groupby('skill_id')['problem_id'].apply(set).apply(list).to_dict()
  #  user_exercises_mapping = df.groupby('user_id')['problem_id'].apply(set).apply(list).to_dict()
    # user_concepts_mapping = df.groupby('user_id')['skill_id'].apply(set).apply(list).to_dict()

    return ChunkMappings(con_id_conv=concepts_id_converter,id_con_conv=id_concepts_converter,exer_id_conv=exercises_id_converter
                         ,exer_con_map=exercise_concepts_mapping,id_exer_conv=id_exercises_converter)
