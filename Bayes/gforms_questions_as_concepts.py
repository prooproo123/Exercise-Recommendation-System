import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import difflib
from sklearn.model_selection import KFold

# Napravi dict koji svakom studentu pridodaje listu savladanosti svih koncepata (gdje se pitanja gledaju kao zasebni koncepti)


def get_student_concept_mastery(df):
  student_concept_mastery={}

  concepts=[column for column in list(df.columns.values) if '[rezultat]' in column]
  i = 0

  for index,row in df.iterrows():
    concept_masteries=[]
    row.apply(lambda cell: concept_masteries.append(1 if cell == '1.00 / 1' else 0))
    student_concept_mastery[i]=concept_masteries
    i+=1
 # print(student_concept_mastery)
  return student_concept_mastery, concepts


def get_px(concepts, student_concept_mastery):
  students=list(student_concept_mastery.keys())
  no_students=len(students)
  no_concepts=len(concepts)
 # print(concepts)
 # print(student_concept_mastery.keys())
 # print(students)

  px= [0.0] * no_concepts
  for i in students:
      for j in range(no_concepts):
        px[j]+=student_concept_mastery[i][j]

  for i in range(no_concepts):
    px[i]/=no_students

  #print(pX)
  return px, no_students, no_concepts


def get_joint_probability(no_concepts, no_students, student_concept_mastery):
  joint_probability=np.zeros((no_concepts,no_concepts))
#  print(student_concept_mastery)
  for i in range(no_students):
    for skill_index1 in range(no_concepts):
      for skill_index2 in range(no_concepts):

        masteries=student_concept_mastery[i]
        if ( masteries[skill_index1] == 1.0 and masteries[skill_index2] == 1.0):
            joint_probability[skill_index1,skill_index2] += 1.0

  for i in range(no_concepts):
    for j in range(no_concepts):
      joint_probability[i,j] /= no_students

# print(joint_probability)
  return joint_probability


# Micanje "[rezultati]" iz imena koncepata

def get_concept_names(concepts):
    concepts = [concept.replace(' [rezultat]', '') for concept in concepts]
    # print(concepts)

    label_dict = {}
    for i in range(0, len(concepts)):
        label_dict[i] = concepts[i]

    return label_dict


def get_dependency_matrix(joint_probability, px):
    dependency_matrix = (np.array(joint_probability)).copy()
    dim = dependency_matrix.shape[0]

    for i in range(0, dim):
        for j in range(0, dim):
            if px[i] != 0:
                dependency_matrix[i, j] /= px[i]
            else:
                dependency_matrix[i, j] = 0
    return dependency_matrix


def get_cutoff(dependency_matrix,cutoff_threshold):
  dim= len(dependency_matrix)
  dependency_matrix = np.subtract(dependency_matrix, np.identity(dim))

  edges = []
  for i in range(dim):
    for j in range(dim):
      if dependency_matrix[i,j] >= cutoff_threshold:
        edges.append((i,j))

  return edges


def draw_graph(edges):
  G = nx.DiGraph()
  G.add_edges_from(edges)

  pos = nx.spring_layout(G)
  plt.gcf().set_size_inches(20, 10)
  nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), node_size = 500)
  nx.draw_networkx_labels(G, pos)
  nx.draw_networkx_edges(G, pos, edgelist=edges, arrows=True)
  plt.show()


def calculate(df):
  student_concept_mastery, concepts = get_student_concept_mastery(df)
  px, no_students, no_concepts = get_px(concepts, student_concept_mastery)
  joint_probability = get_joint_probability(no_concepts, no_students, student_concept_mastery)
  dependency_matrix = get_dependency_matrix(joint_probability,px)
  return dependency_matrix, concepts


def build_graph(df, cutoff = 0.8, kfold = False):

  print("originalni graf:")
  dependency_matrix, concepts = calculate(df)
  label_dict = get_concept_names(concepts)

  edges_org = get_cutoff(dependency_matrix,cutoff)
  draw_graph(edges_org)
  print(len(edges_org))

  if kfold:
    # k-fold cross valiation
    kf = KFold(n_splits = 5, shuffle = True, random_state = 2)
    for train_index, test_index in kf.split(df):
      train, test = df.iloc[train_index], df.iloc[test_index]

    #for i in range(0,10):
    # za random 80%
    #  df2 = df.sample(n = int(len(df)*0.8))

      dependency_matrix, concepts = calculate(train)
      edges = get_cutoff(dependency_matrix,cutoff)
      draw_graph(edges)
      print(len(edges))
      sm = difflib.SequenceMatcher(None,edges_org,edges)
      print(sm.ratio())
      print()

  for i in label_dict:
    print(str(i) + ": " + label_dict[i])