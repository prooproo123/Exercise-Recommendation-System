import pandas as pd
import random


def standard(listt):
    return str(listt).replace(" ", "").replace("[", "").replace("]", "")

def make_csv(df,student_list,filename,path,exercises_id_converter,train=True):
  suffix= '_train1.csv' if train else '_valid1.csv'
  f=open(path+filename+suffix,"w+")
  for student in student_list:
    questions=[]
    answers=[]
    cond=df['user_id'] == student
    temp_df=df[cond]
    temp_df.head()
    #print(len(temp_df))
    questions =[exercises_id_converter[i] for i in temp_df['problem_id'].tolist()]
    answers=temp_df['correct'].tolist()
    no_questions=len(questions)
    f.write(str(no_questions)+'\n')
    f.write(standard(questions)+'\n')
    f.write(standard(answers)+'\n')

  f.close()

def make_variable(df,student_list,exercises_id_converter,train=True):
    variable=''

    for student in student_list:
        #cond = df['user_id'] == student
        temp_df = df[df['user_id'] == student]
        questions =[exercises_id_converter[i] for i in temp_df['problem_id'].tolist()]
        answers = temp_df['correct'].tolist()
        no_questions = len(questions)
        variable += str(no_questions) +'\n'
        variable += standard(questions) +'\n'
        variable += standard(answers) +'\n'

    return variable

def create_from_file(filename,path,exercises_id_converter,csv=True,train_part=0.7):
    #Udio training seta od cijelog dataseta

    df=pd.read_csv(path+filename+'.csv',index_col=None,delimiter='\t')
    return create_from_dataframe(filename,df,exercises_id_converter,csv,train_part,path)


def create_from_dataframe(df,exercises_id_converter,filename=None,csv=True,train_part=0.7,path=''):
    students = df.user_id.unique()
    random.shuffle(students)

    train_size = round(len(students) * train_part)
    # print(train_size)
    train_students = students[:train_size]
    # print(len(train_students))
    validation_students = students[train_size:-1]
    # print(len(validation_students))
    if csv == True:
        make_csv(df, train_students, filename, path,exercises_id_converter)
        make_csv(df, validation_students, filename, path, exercises_id_converter,train=False)
    else:
        return make_variable(df, train_students,exercises_id_converter), make_variable(df, validation_students,exercises_id_converter, train= False)
