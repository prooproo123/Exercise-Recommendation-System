import pandas as pd

'''
biologija=pd.read_csv('C:/Users/Admin/Downloads/Dataset.csv',sep='\t',index_col=None,
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

biologija=biologija[biologija['skill_id'].notnull()]
biologija=biologija[biologija['skill_id'].notna()]

'''
assistments=pd.read_csv('data/skill_builder/skill_builder_data.csv',index_col=None,
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

df1 = assistments[assistments.isna().any(axis=1)]
print(len(df1),len(assistments))

df2=assistments[assistments.notna().all(axis=1)]
print(len(df2))
if len(assistments) - len(df1) == len(df2):
    print('success')


'''
biologiji ne smetaju .notnull(), .notna() varijable 

'''
'''
Dodavanje kt_algos_master u path, za prepare bio dodati import "as"
Staviti  fromVariable u new_kt


trace=[(1, 0), (27, 1)]
candidate_exercises=[3, 7, 9, 13, 15, 17, 18, 25, 26, 29]
a2i = dict(zip(candidate_exercises, range(len(candidate_exercises))))
trace = [(a2i[i[0]], i[1]) for i in trace]
print(a2i)
'''

a=[1,1,1,2,4,6,8,8]
a=list(set(a))
print(a)