import pandas as pd

path='../data/skill_builder/skill_builder_data.csv'

TRAIN_PART=0.8
TEST_PART=1-TRAIN_PART

df=pd.read_csv(path,dtype={'order_id': int, 'assignment_id': int, 'user_id': int, 'assistment_id': int,
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
                                  'first_action', 'opportunity', ])

new_name=path.replace('.csv', '_')

train_rows=round(TRAIN_PART*len(df))
test_rows=len(df)-train_rows

df_train=df[:train_rows]
df_test=df[train_rows:]

df_train.to_csv(new_name+'sakt_train.csv')
df_test.to_csv(new_name+'sakt_test.csv')
