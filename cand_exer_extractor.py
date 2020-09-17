import numpy as np
'''
Inicjalizira se sa attention matricom
Treshold function je funkcija koja prima redak i po nekom uvjetu vraca samo indekse onih elemenata koji su odabrani
,ineksi elemenata koji su 0 ne mogu biti odabrani

Normalize function se obavlja nakon svakog izvlacenja kandidata iz retka,
npr. treshold-percentile, normalization-softmax

'''

class PersonalCandidates:

    def __init__(self, attention_matrix,treshold_function,treshold,normalization_function):
        self.relevancy_matrix=attention_matrix
        self.relevancy_matrix+=self.relevancy_matrix.transpose()
        print(self.relevancy_matrix)
        self.treshold_function=treshold_function
        self.normalization_function=normalization_function
        self.treshold=treshold

    #Primaju se zadaci koje je korisnik prethodno radio
    def get_candidates(self,student_traces):
        candidates=set()

        #Prodi sve retke i izvuci kandidate s obzirom na treshold funkciju
        for trace in student_traces:
            # Sve elemente matrice ciji su indeksi tracevi zadataka koji su odradeni treba postaviti u nulu
            assert isinstance(trace, int)
            row = self.relevancy_matrix[trace]
            row = self.normalize(row)
            row = [row[i] if i not in student_traces else 0 for i in range(len(row))]

           # print(row)
           # print(row)
            self.relevancy_matrix[trace] = row
            app=self.treshold_function(self.relevancy_matrix[trace],treshold=self.treshold)
            print(app)
            candidates=candidates | set(app)
#           candidates.add(app)
        if len(candidates) == 0: #nema vise za preporuciti
            return -1
        return list(candidates)

    #poziva se na pocetku i nakon svakog preporucivanja/ stavljanja nove matrice
    def normalize(self,row):
        #for i in range(len(self.relevancy_matrix)):
          #  self.relevancy_matrix[i]=self.normalization_function(self.relevancy_matrix[i])
        return self.normalization_function(row)


#Normalizacije
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

#ignorira elemente koji imaju nula te radi softmax nad onima koji imaju neku vrijednost
def selective_softmax(x):
    non_zero_indices=[i for i in range(len(x)) if x[i] != 0 ]
   # print(non_zero_indices)
    non_zero_values=[x[i] for i in non_zero_indices]
   # print(non_zero_values)
    non_zero_values=softmax(non_zero_values)
   # print(non_zero_values)
    non_zero_values=[non_zero_values[i] if i in non_zero_indices else 0 for i in range(len(x))]
   # print(non_zero_values)
    return non_zero_values

def divide_by_largest(x):
    return x / max(x)

def no_normalization(x):
    return x

#Pragovi
def percentile_treshold(row,treshold=0.9):#problem ako je malo zadataka i puno ih ima nulu
   # print(row)
    treshold=np.percentile(row,treshold,axis=0)
   # print("Treshold je "+str(treshold))
    return constant_treshold(row,treshold)

def constant_treshold(row,treshold=0.5):
    new = []
    for i in range(len(row)):
        if row[i] >= treshold:
            new.append(i)
   # print(new)
    return new

def min_number_of_exercises(row,treshold=1):
    if len(row) <= min:
        return [i for i in range(len(row)) if i != 0]
    srtd=sorted(row)
    srtd.reverse()
    srtd=srtd[:min]
    exercises=[]
    for index,element in enumerate(row):
        if(element in srtd):
            exercises.append(index)
    return exercises

def no_treshold(row):
    return [i for i in range(len(row))]

'''
Proba
attention= np.array([[0.5,0,0],[0.3,0.2,0],[0.15,0.17,0.2]])
'''

attention= np.array([[0.5,0,0],[0.3,0.2,0.0],[0.15,0.17,0.2]])
treshold=0.4
personal=PersonalCandidates(attention,constant_treshold,treshold,selective_softmax)
print(personal.get_candidates([1,2]))

'''
Ideja je da svaki korisnik ima svojeg candidates exercisesa,
problemi- sto ako se zeli staviti novi relevancy_matrix - jednostavnije samo se sve resetira (ne uzimaju se u obzir prosli tracevi)
        -
'''
