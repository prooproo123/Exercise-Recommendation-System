import numpy as np
'''
Inicjalizira se sa attention matricom
Treshold function je funkcija koja prima redak i po nekom uvjetu vraca samo indekse onih elemenata koji su odabrani
,ineksi elemenata koji su 0 ne mogu biti odabrani

Normalize function se obavlja nakon svakog izvlacenja kandidata iz retka,
npr. treshold-percentile, normalization-softmax

RUBNI SLUCAJ- sto ako je student traces prazan?
            -ako se nema sto za preporuciti vraca se -1?

'''

class PersonalCandidates:

    def __init__(self, attention_matrix,treshold_function,treshold,normalization_function):
        self.relevancy_matrix=attention_matrix
        self.relevancy_matrix+=self.relevancy_matrix.transpose()
      #  print(self.relevancy_matrix)
        self.treshold_function=treshold_function
        self.normalization_function=normalization_function
        self.treshold=treshold

    #Primaju se zadaci koje je korisnik prethodno radio
    def get_candidates(self,student_traces):

        candidates=set()
        #Prodi sve retke i izvuci kandidate s obzirom na treshold funkciju
        for index in range(len(self.relevancy_matrix[0])):
            # Sve elemente matrice ciji su indeksi tracevi zadataka koji su odradeni treba postaviti u nulu
            assert isinstance(index, int)
            row = self.relevancy_matrix[index]
            row = [row[i] if i not in student_traces else 0 for i in range(len(row))]
            if index in student_traces:
              #  print(row)
                row = self.normalize(row)
               # print(row)
                self.relevancy_matrix[index] = row
                app=self.treshold_function(self.relevancy_matrix[index],treshold=self.treshold)
              #  print(app)
                candidates=candidates | set(app)
    #           candidates.add(app)
        if len(candidates) == 0: #nema vise za preporuciti
            return []
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
   # non_zero_values=[non_zero_values.pop(0) if i in non_zero_indices else 0 for i in range(len(x))] ndarray nema pop
    for i in range(len(x)):
        if i in non_zero_indices:
            if len(non_zero_values)==1:
                x[i]=non_zero_values[0]
            else:
                x[i],non_zero_values=non_zero_values[0],non_zero_values[1:]
        else:
            x[i]=0
   # print(non_zero_values)
    return x

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

#maksimalni broj preporucenih zadataka po jednom rijesenom zadataku, za ovo ne treba nikakva normalizacija
def max_number_of_exercises(row,treshold=1):
    if len(row) <= treshold:
        return [i for i in range(len(row)) if row[i] != 0]
    srtd=sorted(row)
    #print(row)
    srtd.reverse()
    #print(srtd)
    #print(list(enumerate(row)))
    srtd=srtd[:treshold]
    #print(srtd)
    exercises=[]
    for index,element in enumerate(row):
        if(element in srtd and element != 0):
            exercises.append(index)
    return exercises

def no_treshold(row):
    return [i for i in range(len(row)) if row[i] != 0]

'''
Proba
attention= np.array([[0.5,0,0],[0.3,0.2,0],[0.15,0.17,0.2]])
attention= np.array([[0.5,0,0,0],[0.3,0.2,0.0,0],[0.15,0.17,0.2,0],[0.772,0.51,0.22,0.11]])
treshold=2
personal=PersonalCandidates(attention,max_number_of_exercises,treshold,selective_softmax)
print(personal.get_candidates([1,2]))
'''