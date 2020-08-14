import numpy as np
from numpy import random
import pandas as pd

#Generates list of answers from all artifial students for all concepts
def get_correctness_list(concept_properties):
  correctness_list = []
  for concept in range(len(concept_properties)):
    no_students, no_questions, mean, st_dev = concept_properties[concept]
    correctness_list += get_students_answers(no_students, no_questions, mean, st_dev)

 # print(correctness_list)
  return correctness_list


#Generates a list of student answers of all students for one concept
def get_students_answers(no_students, no_questions, mean, st_dev):

  knowledge_distribution = random.normal(loc=mean * no_questions, scale=st_dev * no_questions, size=no_students)
  students_answers = []

  for student in range(0, no_students):
    percentage = knowledge_distribution[student]

    no_correct_questions = int(round(percentage))

    answers = [1] * no_questions if no_correct_questions >= no_questions else [0] * (no_questions - no_correct_questions) + [1] * no_correct_questions
    random.shuffle(answers)
    students_answers += answers

 # print(students_answers)
  return students_answers


#Artificial dataset generation, args-(conceptProperties - list of 4 element tuples, where elements represent:number of students,number of questions,correct answers mean,standard deviation respectively)
def create_dataset(concept_properties):
    no_concepts = len(concept_properties)
    no_all_questions = sum([i[1] for i in concept_properties])
    no_ids = 0

    for concept in concept_properties:
        no_ids += concept[0] * concept[1]

    question_id_list = [i for i in range(no_ids)]

    student_list = [i for concept_property in concept_properties for i in range(concept_properties[0]) for j in
                    range(concept_property[1])]

    skill_list = [str(index) for index in range(len(concept_properties)) for i in
                  range(concept_properties[index][1] * concept_properties[index][0])]

    correctness_list = get_correctness_list(concept_properties)

    #  print(len(question_id_list),end='\n')
    #  print(len(student_list),end='\n')
    #  print(len(skill_list),end='\n')
    #  print(len(correctness_list),end='\n')

    output_data = {'num': question_id_list, 'student': student_list, 'skill': skill_list, 'right': correctness_list}
    output_dataframe = pd.DataFrame(output_data)
    output_dataframe.to_csv('ArtificialDataset.csv', sep='\t', encoding='utf-8', index=False)

  #  if download:
    #    files.download('ArtificialDataset.csv')
    return output_dataframe