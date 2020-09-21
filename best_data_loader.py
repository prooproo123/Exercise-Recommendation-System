import pickle
import pandas as pd
import numpy as np
import os

#'/home/zvonimir/Exercise-Recommendation-System/data/skill_builder_pickle.pkl'
PATH_TO_EXREC='/home/zvonimir/'

class Data_Loader():
    def __init__(self, n_questions=100, seqlen=150, seperate_char=','):
        # assist2009 : seq_len(200), n_questions(110)
        # Each value is seperated by seperate_char
        self.seperate_char = seperate_char
        self.n_questions = n_questions
        self.seq_len = seqlen
        self.data_path=os.path.curdir

    '''
    Data format a followed
    1) Number of exercies
    2) Exercise tag
    3) Answers
    '''
    def load_data2(self,data):
        q_data = list()
        qa_data = list()
        print(data)
        for lineid, line in enumerate(data.split('\n')[:-1]):
            # strip
            line = line.strip()
            # Exercise tag line
            if lineid % 3 == 1:
                # split by ',', returns tag list
                print
                'Excercies tag',
                q_tag_list = line.split(self.seperate_char)

            # Answer
            elif lineid % 3 == 2:
                print(', Answers')
                answer_list = line.split(self.seperate_char)

                # Divide case by seq_len
                if len(q_tag_list) > self.seq_len:
                    n_split = len(q_tag_list) // self.seq_len
                    if len(q_tag_list) % self.seq_len:
                        n_split += 1
                else:
                    n_split = 1
                print('Number of split : %d' % n_split)

                # Contain as many as seq_len, then contain remainder
                for k in range(n_split):
                    q_container = list()
                    qa_container = list()
                    # Less than 'seq_len' element remained
                    if k == n_split - 1:
                        end_index = len(answer_list)
                    else:
                        end_index = (k + 1) * self.seq_len
                    for i in range(k * self.seq_len, end_index):
                        # answers in {0,1}
                        qa_values = int(q_tag_list[i]) + int(answer_list[i]) * self.n_questions
                        q_container.append(int(q_tag_list[i]))
                        qa_container.append(qa_values)
                        print('Question tag : %s, Answer : %s, QA : %s' % (q_tag_list[i], answer_list[i], qa_values))
                    # List of list(seq_len, seq_len, seq_len, less than seq_len, seq_len, seq_len...
                    q_data.append(q_container)
                    qa_data.append(qa_container)

        # Convert it to numpy array
        q_data_array = np.zeros((len(q_data), self.seq_len))
        for i in range(len(q_data)):
            data = q_data[i]
            # if q_data[i] less than seq_len, remainder would be 0
            q_data_array[i, :len(data)] = data

        qa_data_array = np.zeros((len(qa_data), self.seq_len))
        for i in range(len(qa_data)):
            data = qa_data[i]
            # if qa_data[i] less than seq_len, remainder would be 0
            qa_data_array[i, :len(data)] = data

        return q_data_array, qa_data_array



    # path : data location
    def load_data(self, path):
        f_data = open(path, 'r')
        # Question/Answer container
        q_data = list()
        qa_data = list()
        # Read data
        for lineid, line in enumerate(f_data):
            # strip
            line = line.strip()
            # Exercise tag line
            if lineid % 3 == 1:
                # split by ',', returns tag list
                print
                'Excercies tag',
                q_tag_list = line.split(self.seperate_char)

            # Answer
            elif lineid % 3 == 2:
                print(', Answers')
                answer_list = line.split(self.seperate_char)

                # Divide case by seq_len
                if len(q_tag_list) > self.seq_len:
                    n_split = len(q_tag_list) // self.seq_len
                    if len(q_tag_list) % self.seq_len:
                        n_split += 1
                else:
                    n_split = 1
                print('Number of split : %d' % n_split)

                # Contain as many as seq_len, then contain remainder
                for k in range(n_split):
                    q_container = list()
                    qa_container = list()
                    # Less than 'seq_len' element remained
                    if k == n_split - 1:
                        end_index = len(answer_list)
                    else:
                        end_index = (k + 1) * self.seq_len
                    for i in range(k * self.seq_len, end_index):
                        # answers in {0,1}
                        qa_values = int(q_tag_list[i]) + int(answer_list[i]) * self.n_questions
                        q_container.append(int(q_tag_list[i]))
                        qa_container.append(qa_values)
                        print('Question tag : %s, Answer : %s, QA : %s' % (q_tag_list[i], answer_list[i], qa_values))
                    # List of list(seq_len, seq_len, seq_len, less than seq_len, seq_len, seq_len...
                    q_data.append(q_container)
                    qa_data.append(qa_container)
        f_data.close()

        # Convert it to numpy array
        q_data_array = np.zeros((len(q_data), self.seq_len))
        for i in range(len(q_data)):
            data = q_data[i]
            # if q_data[i] less than seq_len, remainder would be 0
            q_data_array[i, :len(data)] = data

        qa_data_array = np.zeros((len(qa_data), self.seq_len))
        for i in range(len(qa_data)):
            data = qa_data[i]
            # if qa_data[i] less than seq_len, remainder would be 0
            qa_data_array[i, :len(data)] = data

        return q_data_array, qa_data_array

    def pickle_candidates(self):
        assistments_pickled = PATH_TO_EXREC+'Exercise-Recommendation-System/data/skill_builder_pickle.pkl'

        with open(assistments_pickled, 'rb') as f:
            df = pickle.load(f)

        exercises = df['problem_id'].unique()
        concepts = df['skill_id'].unique()

        concepts_exercises_dict = dict.fromkeys(concepts)
        exercises_concepts_dict = dict.fromkeys(exercises)

        for concept in concepts:
            condition2 = df['skill_id'] == concept
            concepts_exercises_dict[concept] = set(df[condition2]['problem_id'])

        for exercise in exercises:
            condition3 = df['problem_id'] == exercise
            exercises_concepts_dict[exercise] = set(df[condition3]['skill_id'])

        # users 64525 and 70363
        # TODO synthesize this and pickle
        student_traces = [[(51424, 1), (51435, 1)], [(51444, 0), (51395, 1), (51481, 0)]]
        student_traces = [[(1, 0), (3, 1)], [(6, 1), (6, 0), (7, 1)]]

        all_candidate_exercises = []
        for trace in student_traces:
            candidate_exercises = set()
            candidate_concepts = set()
            for exercise, answer in trace:
                candidate_concepts.update(exercises_concepts_dict[exercise])
            candidate_concepts = list(candidate_concepts)
            for c in candidate_concepts:
                candidate_exercises.update(concepts_exercises_dict[c])
            all_candidate_exercises.append(list(candidate_exercises))

        with open('data/skill_builder/candidates_pickled.pkl', 'wb') as f:
            pickle.dump(all_candidate_exercises, f)

        new_list = []
        with open('data/skill_builder/candidates_pickled.pkl', 'rb') as f:
            new_list = pickle.load(f)

        a = 0
        return 0

    def load_ids(self):
        pat = PATH_TO_EXREC+'Exercise-Recommendation-System/data/assist2015/assist2015_qname_qid'
        with open(file=pat) as f:
            lines = f.readlines()

        assistments_pickled = PATH_TO_EXREC+'Exercise-Recommendation-System/data/skill_builder_pickle.pkl'
        with open(assistments_pickled, 'rb') as f:
            df = pickle.load(f)

        name_id_ex_dict = {int(line.split()[0]): int(line.split()[1]) for line in lines}

        exercise_concepts_dict = dict.fromkeys(name_id_ex_dict.keys())
        v = np.unique(df[['problem_id', 'skill_id']].values, axis=0).astype(np.int)

        # Knowledge Concepts Corresponding to the exercise
        with open('data/skill_builder/old_e2c.pkl', 'rb') as f:
            q2kg = pickle.dump(exercise_concepts_dict, f)

        return 0

    def standard(self,listt):
        return str(listt).replace(" ", "").replace("[", "").replace("]", "")

    def standardCon(self,listt, di):
        listtt = [di[ex] for ex in listt]
        return str(list(map(int, listtt))).replace(" ", "").replace("[", "").replace("]", "")

    def obradi_assistments1(self):

        assistments_data_path = '/data/skill_builder/skill_builder_data.csv'

        df = pd.read_csv(assistments_data_path,
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
                                  'first_action', 'opportunity', ])

        assistments_pickled = PATH_TO_EXREC+'Exercise-Recommendation-System/data/skill_builder_pickle.pkl'

        students = df.user_id.unique()

        outputFilePath = PATH_TO_EXREC+'Exercise-Recommendation-System/data/stan_con.csv'
        self.for_students_write(self,outputFilePath,df)

    def obradi_assistments2(self):
        assistments_data_path = '/data/skill_builder/skill_builder_data.csv'

        df = pd.read_csv(assistments_data_path,
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
                                  'first_action', 'opportunity', ])

        assistments_pickled = PATH_TO_EXREC+'Exercise-Recommendation-System/data/skill_builder_pickle.pkl'

        students = df.user_id.unique()

        outputFilePath = PATH_TO_EXREC+'Exercise-Recommendation-System/data/stand_exind_nocon.csv'
        self.for_students_write(self,outputFilePath,df)


    def preprocess_gforms(self,filename):
        csv_path=PATH_TO_EXREC+'Exercise-Recommendation-System/data/gforms_raw/'+filename
        df=pd.read_csv(filepath_or_buffer=csv_path,delimiter='\t')

        self.for_students_write(PATH_TO_EXREC+'Exercise-Recommendation-System/data/gforms_raw/'+filename+'processed.csv',df)

    def for_students_write(self,path,df):

        outputFile = open(path, "w+")
        students = df.user_id.unique()
        max_exercises = 0
        max_concepts = 0
        l = []

        ex_ids = df['problem_id'].unique()
        exercises_sorted = sorted(ex_ids)
        exercise_index_mapping = {s: i for i, s in enumerate(exercises_sorted)}

        for student in students:
            condition = df['user_id'] == student
            studentData = df[condition]
            exercises = studentData['problem_id'].values
            answers = studentData['correct'].values
            concepts = studentData['skill_id'].values
            con = np.isnan(concepts)
            concepts = concepts[~con]
            exercises = exercises[~con]
            # time = studentData['skill_id'].tolist()
            # difficulty = studentData['skill_id'].tolist()
            # gate = studentData['skill_id'].tolist()
            exercises = exercises.tolist()
            answers = answers.tolist()
            concepts = concepts.tolist()

            if len(exercises) > max_exercises:
                max_exercises = len(exercises)

            if len(concepts) > max_concepts:
                max_concepts = len(concepts)

            if len(exercises) != len(concepts):
                print("razlika je" + str(len(exercises) - len(concepts)))

            # if len(exercises) < 1900:
            l.append(len(exercises))
            outputFile.write(str(len(exercises)) + "\n")
            outputFile.write(self.standardCon(exercises, exercise_index_mapping) + "\n")
            outputFile.write(self.standard(answers) + "\n")
            # outputFile.write(standardCon(concepts) + "\n")
        # [8214, 6577, 4290, 3263, 2800, 2457, 2354, 2300, 1893, 1849, 1783, 1684, 1674, 1611, 1606, 1565, 1546, 1490, 1482, 1473, 1456, 1442, 1434,...
        l2 = sorted(l, reverse=True)
        outputFile.close()