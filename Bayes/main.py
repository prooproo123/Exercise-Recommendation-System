import gforms_to_dataframe as gformprocess
import artificial_dataset_generator as datagenerator
import bkt_parameter_estimation as paramestimation
import gforms_questions_as_concepts as questionasconcepts
import bayes_graph_generator as graphgen

import pandas as pd
#Graph generation, args-(dataset_parameters, dataset, cutoff, bkt_threshold, gauss_threshold)
def build_graph(parameters, data, cutoff=0.8, bkt_threshold=0.95, gauss_threshold=0.9):
  graphgen.build_graph(parameters, data, cutoff, bkt_threshold, gauss_threshold)

#Artificial dataset generation, args-(conceptProperties - list of 4 element tuples, where elements represent:number of students,number of questions,correct answers mean,standard deviation respectively)
def generate_and_draw(conceptProperties):
 return datagenerator.create_dataset(conceptProperties)

#Calculates BKT parameters for given dataset using simulated annealing method
def estimate_parameters(data):
  return paramestimation.estimate_parameters(data)

# Builds a graph by treating questions as standalone concepts
def questions_to_concepts(data, cutoff = 0.8, kfold = False):
  return questionasconcepts.build_graph(data, cutoff, kfold)

# Converts google forms .csv file to appropriate data format
def form_to_dataset(data, skill_names):
  return gformprocess.process_data(data, skill_names)

#main
#====================================

input_path='C:/Users/Admin/Downloads/Biology quiz.csv'
csv=open("C:/Users/Admin/Downloads/Biology quiz.csv","rb")
skill_names=['Stanica', 'Stanični metabolizam', 'Živčana stanica', 'Dioba stanice', 'DNA']

df2=form_to_dataset(csv,skill_names)
print(df2['problem_id'])
#====================================