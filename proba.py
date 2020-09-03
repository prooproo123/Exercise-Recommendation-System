import pandas as pd


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