import pandas as pd

def split(path,train_part=0.8,sep=','):

    TRAIN_PART=train_part
    TEST_PART=1-TRAIN_PART

    df=pd.read_csv(path,index_col=None,
                        dtype={'user_id': int,
                              'problem_id': int,
                              'correct': int,
                              'skill_id': str,
                              },
                       usecols=['user_id',
                                'problem_id',
                                'correct',
                                'skill_id',
                                ],sep=sep)

    new_name=path.replace('.csv', '_')

    train_rows=round(TRAIN_PART*len(df))
    test_rows=len(df)-train_rows

    df_train=df[:train_rows]
    df_test=df[train_rows:]

    df_train.to_csv(new_name+'sakt_train.csv')
    df_test.to_csv(new_name+'sakt_test.csv')
