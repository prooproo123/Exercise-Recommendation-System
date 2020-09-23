import pickle

import pandas as pd
import numpy as np
from statistics import mean, median
# DEBUGGING PKL FILES

# sb09 = pd.read_csv('/../data/skill_builder/skill_builder_data.csv')

class ChunkInfo:

    def __init__(self,chunk):

        self.users = chunk['user_id'].unique()
        self.exercises = chunk['problem_id'].unique()
        self.concepts = chunk['skill_id'].unique()

        #Mozda nije tocno implementirano, za sada nema jos potrebe za tim vrijednostima
        self.concept_exercises_mapping = chunk.groupby('skill_id')['problem_id'].apply(set).apply(list).to_dict()
        self.user_exercises_mapping = chunk.groupby('user_id')['problem_id'].apply(set).apply(list).to_dict()
        self.user_concepts_mapping = chunk.groupby('user_id')['skill_id'].apply(set).apply(list).to_dict()

        self.concepts_id_converter = {c: i for i, c in enumerate(self.concepts)}

        self.id_concepts_converter = {i: c for i, c in enumerate(self.concepts)}

        self.exercises_id_converter = {e: i for i, e in enumerate(self.exercises)}
        self.id_exercises_converter = {i: e for i, e in enumerate(self.exercises)}

        e = chunk.groupby('problem_id')['skill_id'].apply(set).apply(list).to_dict()

        self.exercise_concepts_mapping = {self.exercises_id_converter[k]: [self.concepts_id_converter[c] for c in v] for k, v in
                                          e.items()}
        self.no_exercises=len(self.exercises)
        self.no_concepts=len(self.concepts)

    def get_no_concepts(self):
        return self.no_concepts

    def get_no_exercises(self):
        return self.no_exercises

    def get_no_users(self):
        return self.no_users

    def get_exercise_concepts_mapping(self):
        return self.exercise_concepts_mapping

    def get_concept_exercises_mapping(self):
        return self.concept_exercises_mapping

    def get_user_exercises_mapping(self):
        return self.user_exercises_mapping

    def get_user_concepts_mapping(self):
        return self.user_concepts_mapping

    def get_concept_id_converter(self):
        return self.concepts_id_converter

    def get_id_concepts_converter(self):
        return self.id_concepts_converter

    def get_exercises_id_converter(self):
        return self.exercises_id_converter

    def get_id_exercise_converter(self):
        return self.id_exercises_converter

def get_chunks(filepath,chunk_size=40000,sep='\t'):
    dataset = pd.read_csv(filepath,sep=sep,index_col=None,
                       dtype={'user_id': int,
                              'problem_id': int,
                              'correct': int,
                              'skill_id': str,
                              },
                       usecols=['user_id',
                                'problem_id',
                                'correct',
                                'skill_id',
                                ])

    dataset=dataset[dataset.notna().all(axis=1)]
    list_df = [dataset[i:i+chunk_size] for i in range(0,dataset.shape[0],chunk_size)]
    return list_df


'''
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

    '''

'''
       print('Number of students: ' + str(len(users)))
       print('Number of exercises: ' + str(len(exercises)))
       print('Number of concepts: ' + str(len(concepts)))
       print()

       print('Maximum exercises per student: ' + str(max([len(vals) for vals in user_exercises_mapping.values()])))
       print('Maximum concepts per student: ' + str(max([len(vals) for vals in user_concepts_mapping.values()])))
       print('Maximum exercises per concept: ' + str(max([len(vals) for vals in concept_exercises_mapping.values()])))
       print('Maximum concepts per exercise: ' + str(max([len(vals) for vals in exercise_concepts_mapping.values()])))
       print()

       print('Median exercises per student: ' + str(median([len(vals) for vals in user_exercises_mapping.values()])))
       print('Median concepts per student: ' + str(median([len(vals) for vals in user_concepts_mapping.values()])))
       print(
           'Median exercises per concept: ' + str(median([len(vals) for vals in concept_exercises_mapping.values()])))
       print(
           'Median concepts per exercise: ' + str(median([len(vals) for vals in exercise_concepts_mapping.values()])))
       print()
       '''