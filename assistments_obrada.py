import pandas as pd
import io


def standard(listt):
    return str(listt).replace(" ", "").replace("[", "").replace("]", "")


assistments_data_path = '/home/zvonimir/Exercise-Recommendation-System/data/skill_builder_data.csv'

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

assistments_pickled = '/home/zvonimir/Exercise-Recommendation-System/data/skill_builder_pickle.pkl'

students = df.user_id.unique()

outputFilePath = "o.csv"
outputFile = open(outputFilePath, "w+")

max_exercises = 0
max_concepts = 0
l=[]
for student in students:
    condition = df['user_id'] == student
    studentData = df[condition]
    exercises = studentData['problem_id'].tolist()
    answers = studentData['correct'].tolist()
    concepts = studentData['skill_id'].tolist()
    # time = studentData['skill_id'].tolist()
    # difficulty = studentData['skill_id'].tolist()
    # gate = studentData['skill_id'].tolist()

    if len(exercises)>max_exercises:
        max_exercises=len(exercises)

    if len(concepts)>max_concepts:
        max_concepts=len(concepts)

    if len(exercises)!= len(concepts):
        print("razlika je"+str(len(exercises)-len(concepts)))

    if len(exercises)<1900:
        l.append(len(exercises))
        outputFile.write(str(len(exercises)) + "\n")
        outputFile.write(standard(exercises) + "\n")
        outputFile.write(standard(answers) + "\n")
        outputFile.write(standard(concepts) + "\n")
#[8214, 6577, 4290, 3263, 2800, 2457, 2354, 2300, 1893, 1849, 1783, 1684, 1674, 1611, 1606, 1565, 1546, 1490, 1482, 1473, 1456, 1442, 1434,...
l2=sorted(l,reverse=True)
outputFile.close()
