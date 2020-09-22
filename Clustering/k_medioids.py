import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
#from google.colab import files
import io
import pandas as pd
from collections import Counter
import math
import itertools as it
import numpy as np


def read_student_answers(df, skills):
    answers_to_questions = []
    real_classes = []

    for skill in skills:
        questions = list(set(df[df['skill_id'] == skill]['problem_id'].tolist()))
        for question in questions:
            real_classes.append(skills.index(skill))
            rows = df[(df['skill_id'] == skill) & (df['problem_id'] == question)]
            answers = []
            for i in range(len(rows)):
                answers.append(rows.iloc[i]['correct'])

            answers_to_questions.append(answers)

    return answers_to_questions, real_classes


def distance_function(x, y):
    difference_sum = 0

    for i in range(len(x)):
        difference_sum += abs(x[i] - y[i])

    return difference_sum


def adjusted_rand_index(real_classes, result):
  return adjusted_rand_score(real_classes, result)


def rand_index(y_true, y_predict):
    indeksi = list(range(0, len(y_true),1))
    kombinacije = list(it.combinations(indeksi, 2))
    a = 0
    b = 0

    for komba in kombinacije:
        if (y_true[komba[0]] == y_true[komba[1]] and y_predict[komba[0]] == y_predict[komba[1]]):
            a += 1
        else:
            if (y_true[komba[0]] != y_true[komba[1]] and y_predict[komba[0]] != y_predict[komba[1]]):
                b += 1
    return (a+b)/len(kombinacije)


def mutual_info_score(real_classes, result):
  return normalized_mutual_info_score(real_classes, result)


def sum_error(real_classes, result):
    sum = 0

    for i in range(0, len(real_classes), 6):
        counter = Counter(result[i:i + 6])
        largest = 0
        for count in counter:
            sum += counter[count]
            if counter[count] > largest:
                largest = counter[count]
        sum -= largest
    return sum


def scale(answers):
  scaler = StandardScaler()
  scaled_answers = scaler.fit_transform(answers)
  #print(scaler.mean_)
  #print(scaled_answers)
  return scaled_answers


def define_k(answers):
  kmeans_kwargs = {
          "init": "random",
          "n_init": 10,
          "max_iter": 300,
          "random_state": None,
          }
  sse = []
  for k in range(1, 11):
      kmeans = KMedoids(n_clusters=k, metric=distance_function, random_state=None).fit(answers)
      kmeans.fit(scaled_answers)
      sse.append(kmeans.inertia_)

  plt.style.use("fivethirtyeight")
  plt.plot(range(1, 11), sse)
  plt.xticks(range(1, 11))
  plt.xlabel("Number of Clusters")
  plt.ylabel("SSE")
  plt.show()
  kl = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")
  kl.elbow
  return kl.elbow


def calculate_clusters(answers, real_classes):
  kmedoids1 = KMedoids(n_clusters=5, metric=distance_function, random_state=None).fit(answers)
  print(kmedoids1.labels_)

  kmedoids2 = KMedoids(n_clusters=5, metric=distance_function, random_state=None).fit(scale(answers))
  print(kmedoids2.labels_)

  print("Rand index:")
  print(rand_index(real_classes, kmedoids1.labels_))
  print(rand_index(real_classes, kmedoids2.labels_))

  print("Adjusted rand index:")
  print(adjusted_rand_index(real_classes, kmedoids1.labels_))
  print(adjusted_rand_index(real_classes, kmedoids2.labels_))

  print("Sum error:")
  print(sum_error(real_classes, kmedoids1.labels_))
  print(sum_error(real_classes, kmedoids2.labels_))

  print("NMI:")
  print(mutual_info_score(real_classes, kmedoids1.labels_))
  print(mutual_info_score(real_classes, kmedoids2.labels_))

def cluster(path_to_dataset):

    df = pd.read_csv(path_to_dataset,delimiter='\t')

    skills = list(set(df['skill_id'].tolist()))
    answers, real_classes = read_student_answers(df, skills)
    calculate_clusters(np.array(answers), real_classes)
    print(real_classes)

    scaled_answers = scale(answers)
    define_k(answers)