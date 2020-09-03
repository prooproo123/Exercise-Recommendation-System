#chunk analysis- treba staviti varijabilni chunk size i ako se stavi preveliki da stane kad dode do zadnjeg
                #treba srediti u koji folder idu rezultati kako bi se dalje moglo citati
import pickle

import chunk_analysis as ca
import tempfile_creator as tc
import train_and_validate_creator as tvc
#import new_kt as kt
import new_rs as rs
import student_traces_extractor as traces


path='data/biology30/biology30.csv'
chunks=ca.get_chunks(path)
print(chunks[0])

info= ca.ChunkInfo(chunks[0])

print(len(chunks))

#chunk_mappings=tc.get_mappings(chunks[0])
exercise_concepts_mapping,exercises_id_converter=info.get_exercise_concepts_mapping(),info.get_exercises_id_converter()

print(exercise_concepts_mapping, end='\n')
print(exercises_id_converter,end='\n')

no_exercises=info.get_no_exercises()
print(no_exercises)
no_concepts=info.get_no_concepts()
print(no_concepts)
#TREBA OD NEGDJE DOBITI LISTU SVIH STUDENT TRACEOVA I CANDIDATE EXERCISES za sada i dalje hardkodirati
'''
tvc.create(filename,path_to_dir)
filename='biology30'
path_to_dir='data/biology30/'
train_variable,valid_variable=tvc.create_from_dataframe(filename,chunks[0],exercise_concepts_mapping,False)
'''

#kt
#params= kt.main(filename,path_to_dir)#za sada se train i valid citaju iz csv-a, trebalo bi skuziti kako ih dobiti ko datafreameove iz skripte za stvaranje
file = open("data/biology30/kt_params",'rb')
params = pickle.load(file)

'''
path_to_gform_traces=''
all_student_traces= traces.get_traces_from_gforms(path_to_gform_traces)
'''
stu=[[(1, 0), (27, 1)]]
cands=[1, 15, 16, 27]
recommendation=rs.run_rs(stu,cands,params,exercise_concepts_mapping,exercises_id_converter,no_exercises,no_concepts)