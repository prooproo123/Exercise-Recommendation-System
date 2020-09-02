import pickle

import pandas as pd
import numpy as np
from statistics import mean, median
# DEBUGGING PKL FILES

# sb09 = pd.read_csv('/../data/skill_builder/skill_builder_data.csv')

class ChunkInfo:

    def __init__(self,chunk):
     # removing rows with nan concept value
        # chunk = chunk[chunk['skill_id'].notna()]

        # chunk['skill_id'] = chunk['skill_id'].astype(np.int64)

        self.users = chunk['user_id'].unique()
        self.exercises = chunk['problem_id'].unique()
        self.concepts = chunk['skill_id'].unique()

        self.exercise_concepts_mapping = chunk.groupby('problem_id')['skill_id'].apply(set).apply(list).to_dict()
        self.concept_exercises_mapping = chunk.groupby('skill_id')['problem_id'].apply(set).apply(list).to_dict()
        self.user_exercises_mapping = chunk.groupby('user_id')['problem_id'].apply(set).apply(list).to_dict()
        self.user_concepts_mapping = chunk.groupby('user_id')['skill_id'].apply(set).apply(list).to_dict()
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


def get_chunks(filepath,chunk_size=40000):
    sb09 = pd.read_csv(filepath,sep='\t',index_col=None)
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
    n = 40000  #chunk row size
    list_df = [sb09[i:i+n] for i in range(0,sb09.shape[0],n)]
    #chunk=list_df[3]
    # with open('data/skill_builder/chunk.pkl', 'wb') as f:
    #     pickle.dump(d,f)
    #chunk = chunk[chunk['skill_id'].notna()] #zasto ovo izbacuje gresku za biologiju

    #chunk['skill_id'] = chunk['skill_id'].astype(np.int64)
    #chunk.to_csv('data/skill_builder/chunk.csv', columns=['user_id', 'problem_id', 'correct', 'skill_id'])


    return list_df
