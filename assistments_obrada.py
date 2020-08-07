import pandas as pd
import io


def standard(listt):
  return str(listt).replace(" ","").replace("[","").replace("]","")

filepath=""
asistments=open(filepath,"r")

students=df.user_id.unique()

outputFilePath=""
outputFile=open(outputFilePath,"w+")

for student in students:
  condition= df['user_id'] == student
  studentData=df[condition]
  exercises=studentData['problem_id'].tolist()
  answers=studentData['correct'].tolist()
  concepts=studentData['skill_id'].tolist()

  outputFile.write(str(len(exercises))+"\n")
  outputFile.write(standard(exercises)+"\n")
  outputFile.write(standard(answers)+"\n")
  outputFile.write(standard(concepts)+"\n")
outputFile.close()


