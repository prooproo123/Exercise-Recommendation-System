import pandas as pd
import random
import numpy as np


def standard(listt):
    return str(listt).replace(" ", "").replace("[", "").replace("]", "")

def make_csv(df,student_list,filename,path,train=True):
  #preffix = 'assist2009_updated'
  suffix= '_train1.csv' if train else '_valid1.csv'

  #dodano za assist
  unique_exercises = df['problem_id'].unique()
  exercises_id_converter = {e: i for i, e in enumerate(unique_exercises)}

  f=open(path+filename+suffix,"w+")
  for student in student_list:
    questions=[]
    answers=[]
    cond=df['user_id'] == student
    temp_df=df[cond]
    temp_df.head()
    questions=temp_df['problem_id'].tolist()
    answers=temp_df['correct'].tolist()
    no_questions=len(questions)

    #assist
    questions = [exercises_id_converter[e] for e in questions]
    
    f.write(str(no_questions)+'\n')
    f.write(standard(questions)+'\n')
    f.write(standard(answers)+'\n')

  f.close()

def create(chunk_filename,filename,path):
    #Udio training seta od cijelog dataseta
    TRAIN_PART=0.7

    #dosad tu bio cijeli skill_builder, ako chunkovi, onda maknuti stupce kojih nema
    #zasad ostavljeni svi stupci
    #df=pd.read_csv(path+'skill_builder_data.csv',sep=',',
    df=pd.read_csv(path+chunk_filename,sep=',')
            #dtype={'user_id': int, 'problem_id': int, 'skill_id': float, correct: int},

            #usecols=['user_id', 'problem_id', 'skill_id', 'correct'])

    students=df.user_id.unique()
    print('students', len(students))
    students_column=df['user_id'].unique()
    print('shape 0', students_column.shape[0])
    random.shuffle(students)

    train_size=round(len(students)*TRAIN_PART)
    #print(train_size)
    train_students=students[:train_size]
    #print(len(train_students))
    validation_students=students[train_size:-1]
    #print(len(validation_students))
    make_csv(df,train_students,filename,path)
    make_csv(df,validation_students,filename,path,False)
