import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore
import io


class BKT:
    def __init__(self, params, skill, threshold):
        self.pL = params[0]
        self.pG = params[1]
        self.pS = params[2]
        self.pT = params[3]
        self.params = params
        self.skill_name = skill
        self.threshold = threshold

    def estimate_pL(self, answers):
        for answer in answers:
            if answer == 1:
                pLnRn = self.pL * (1 - self.pS) / (self.pL * (1 - self.pS) + (1 - self.pL) * self.pG)
            else:
                pLnRn = self.pL * self.pS / (self.pL * self.pS + (1 - self.pL) * (1 - self.pG))

            self.pL = pLnRn + (1 - pLnRn) * self.pT

    def passed(self):
        return 1 if self.pL >= self.threshold else 0

    def get_pL(self):
        return self.pL

    def get_pC(self):
        return self.pL * (1 - self.pS) + (1 - self.pL) * self.pG

    def get_skill_name(self):
        return self.skill_name

    def copy(self):
        return BKT(self.params, self.skill_name, self.threshold)

    def reset(self):
        self.pL = self.params[0]

    def get_threshold(self):
        return self.threshold


class Gauss:
    def __init__(self, student_answers, threshold):
        self.scores = [sum(answer) for answer in student_answers]
        self.threshold = threshold

    def get_percentile(self, student_answer):
        student_score = sum(student_answer)
        self.percentile = percentileofscore(self.scores, student_score)
        return self.percentile / 100

    def passed(self):
        return 1 if self.percentile / 100 > self.threshold else 0

    def get_threshold(self):
        return self.threshold


# Stvaranje dictionaryja (student,skill) = [lista odgovora]

def read_student_answers(df, students, skills):
    student_answers = dict()

    for skill in skills:
        for student in students:
            if ((df['student'] == student) & (df['skill'] == skill)).any():
                answers = []
                rows_df = df[(df['student'] == student) & (df['skill'] == skill)]
                for i in range(len(rows_df)):
                    answers.append(rows_df.iloc[i]['right'])

                student_answers[(student, skill)] = answers

    return student_answers


# Račun za p(X)

def get_skill_mastery_probability(skills, students, skill_student_success_dict):  # izbrisan df argument
    px = []
    for skill in skills:
        #   print(skill)
        average_skill_mastery = 0
        noStudents = 0

        for student in students:
            # bkt = BKT_dict[skill]
            # bkt.estimate_pL(student_answers[(student,skill)])
            # gauss = Gauss_dict[skill]
            # print(gauss_plus_bkt_pass_condition(bkt, gauss, student_answers[(student,skill)]))

            if (student, skill) in skill_student_success_dict:
                noStudents += 1
                average_skill_mastery += skill_student_success_dict[
                    (student, skill)]  # gauss_plus_bkt_pass_condition(bkt, gauss, student_answers[(student,skill)])
        # answers=df[(df['skill'] == skill) & (df['student'] == student)]
        #  correct_answers=answers[answers['right'] == 1]
        # print(answers.shape[0])
        # student_result=correct_answers.shape[0] / answers.shape[0] # broj točnih/broj odgovora
        # average_skill_mastery+=student_result
        # bkt.reset()
        # uprosječuje se uspjeh svih studenata i dodaje u listu
        average_skill_mastery /= noStudents
        px.append(average_skill_mastery)

    noSkills = len(px)
    #  print("pX")
    #  print(pX)
    return px

# BKT parametri svih vještina

def get_skill_params(bkt_param_df):
  skill_params={}
  for i in range(len(bkt_param_df)):
    row=bkt_param_df.iloc[i].tolist()
    skill_params[row[0]]=row[1:]
  return skill_params

# Stvaranje bkt-a za svaki skill i spremanje u dict

def get_bkt_dict(skills,skill_params,bkt_threshold):
  BKT_dict={}
  for skill in skills:
    BKT_dict[skill]=BKT(skill_params[skill], skill, bkt_threshold) #inicijalizirati novi bkt sa parametrima iz nekog filea
  return BKT_dict

# Stvaranje gaussa za svaki skill i spremanje u dict

def get_gauss_dict(skills,students,student_answers,gauss_threshold):
  Gauss_dict={}
  for skill in skills:
    all_answers = []
    for student in students:
      if (student,skill) in student_answers:
        all_answers.append(student_answers[(student,skill)])

    Gauss_dict[skill]=Gauss(all_answers, gauss_threshold) #inicijalizirati novi gauss
  return Gauss_dict


# Ubaciti u bkt svakog studenta
# U dictionary čiji je key tuple (student,vještina) ubaci 0 ili 1 ovisno o tome je li student položio npr. skill_student_success_dict
# Vjerojatnost prolaska ove vještine je suma jedinica iz skill_student_success_dict / len(skill_student_success_dict)

def get_student_success(students,skills,BKT_dict,Gauss_dict,student_answers):
  skill_student_success_dict = dict()

  for student in students:
    for skill in skills:
      if (student,skill) in student_answers:
        bkt=BKT_dict[skill]
        gauss=Gauss_dict[skill]
        skill_student_success_dict[(student,skill)] = gauss_plus_bkt_pass_condition(bkt,gauss,student_answers[(student,skill)])
        bkt.reset()
        #bkt = BKT_dict[skill].copy()
      # bkt.estimate_pL(student_answers[(student,skill)])
        #skill_student_success_dict[(student,skill)] = bkt.passed()
  return skill_student_success_dict

#je li bolje raditi copy ili resetirati bkt na L0 svaki put?

# Ovaj dio koda treba uz pomoć bkt-a i gaussa odrediti je li student položio ili pao

def gauss_plus_bkt_pass_condition(bkt, gauss, answers):
    bkt.estimate_pL(answers)
    bkt_pass = bkt.passed()
    # provjera je li problem u gaussu
    #  return bkt_pass
    gauss.get_percentile(answers)
    gauss_pass = gauss.passed()
    if bkt_pass + gauss_pass == 2:
        return 1
    if bkt_pass + gauss_pass == 0:
        return 0

    bkt_threshold = bkt.get_threshold()
    gauss_threshold = gauss.get_threshold()

    bkt_pL = bkt.get_pL()
    gauss_percentile = gauss.get_percentile(answers)

    bkt_certainty = (bkt_pL - bkt_threshold) / (1 - bkt_threshold) if bkt_pass == 1 else -1 * (
                bkt_threshold - bkt_pL) / (bkt_threshold)
    gauss_certainty = (gauss_percentile - gauss_threshold) / (1 - gauss_threshold) if gauss_pass == 1 else -1 * (
                gauss_threshold - gauss_percentile) / (gauss_threshold)

    #  print(bkt_certainty,gauss_certainty)
    return 1 if bkt_certainty + gauss_certainty > 0 else 0


# Izgradnja matrice p(Xi and Xj)
# Stvoriti matricu odgovarajuće veličine

def get_joint_probability_matrix(skills, students, skill_student_success_dict):
    joint_probability = []

    for skill1 in skills:
        # stvaranje novog retka u matrici za vještinu skill1
        skill1_list = []

        for skill2 in skills:
            skill2_sum = 0.
            noStudents = 0

            for student in students:
                # ako je student položio (vidi se iz prije generiranog dicta) oba predmeta u matricu se na odgovarajućoj poziciji dodaje +1

                if (student, skill1) in skill_student_success_dict and (student, skill2) in skill_student_success_dict:
                    noStudents += 1
                    student_skill1 = skill_student_success_dict[(student, skill1)]
                    student_skill2 = skill_student_success_dict[(student, skill2)]

                    if student_skill1 == 1 and student_skill2 == 1:
                        skill2_sum = skill2_sum + 1.

            # Svaki element matrice podijeliti s brojem studenata
            if noStudents > 0:
                skill1_list.append(float(skill2_sum) / noStudents)
            else:
                skill1_list.append(0)

        joint_probability.append(skill1_list)

    return joint_probability


# Dependency matrix generation p(Xi|Xj)
# row Xi-Y, column Xj-X

def get_dependency_matrix(joint_probability, px):
    dependency_matrix = (np.array(joint_probability)).copy()
    dim = dependency_matrix.shape[0]

    for i in range(0, dim):
        for j in range(0, dim):
            dependency_matrix[i, j] /= px[i]
    return dependency_matrix

#Crtanje grafa

def draw_graph(edges):
  G = nx.DiGraph()
  G.add_edges_from(edges)

  pos = nx.spring_layout(G)
  nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), node_size = 500)
  nx.draw_networkx_labels(G, pos)
  nx.draw_networkx_edges(G, pos, edgelist=edges, arrows=True)
  plt.show()

#Cutoff i micanje dijagonale

def do_cutoff(dependency_matrix,cutoff_threshold):
  dim= len(dependency_matrix)
  dependency_matrix = np.subtract(dependency_matrix, np.identity(dim))

  edges = []
  for i in range(dim):
    for j in range(dim):
      if dependency_matrix[i,j] >= cutoff_threshold:
        edges.append((i,j))

  return edges

def build_graph(skill_params, df, cutoff = 0.8, bkt_threshold = 0.95, gauss_threshold = 0.9):

  skills = list(set(df['skill'].tolist()))
  students = list(set(df['student'].tolist()))
  student_answers = read_student_answers(df, students, skills)
  bkt_dict = get_bkt_dict(skills,skill_params,bkt_threshold)
  gauss_dict = get_gauss_dict(skills,students,student_answers,gauss_threshold)
  skill_student_success_dict = get_student_success(students,skills,bkt_dict,gauss_dict,student_answers)
  px = get_skill_mastery_probability(skills,students,skill_student_success_dict)
  joint_probability = get_joint_probability_matrix(skills,students,skill_student_success_dict)
#  print("joint")
#  print(joint_probability)
  dependency_matrix = get_dependency_matrix(joint_probability,px)
#  print("dependency")
#  print(dependency_matrix)

 # CUTOFF_THRESHOLD=0.5

  edges = do_cutoff(dependency_matrix, cutoff)
  print(skills)
  draw_graph(edges)


