import pandas as pd
import chunk_analysis as ca

#za svakog studenta treba izvuci trojku (num_q, problems (valjda mapirani id-ovi),correct)



def extract_students_from_csv(path_to_csv,sep='\t'):
    df=pd.read_csv(path_to_csv,sep=sep,index_col=None,
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

    analysis=ca.ChunkInfo(df)
    return extract_students_from_df(df,analysis.get_exercises_id_converter())

def extract_students_from_df(df,exercises_id_converter):
    student_tuples=[]
    students = df.user_id.unique()
    #treba provjeriti je li to dobar format u kojem se salju podatci
    for student in students:
        cond = df['user_id'] == student
        temp_df = df[cond]
        temp_df.head()
        # print(len(temp_df))
        questions = [exercises_id_converter[i] for i in temp_df['problem_id'].tolist()]
        answers = temp_df['correct'].tolist()
       # print("STUDENT IMA "+str(len(answers))+" odgovora")
        student_tuples.append((len(answers),questions,answers))

    return len(df.problem_id.unique()),student_tuples
