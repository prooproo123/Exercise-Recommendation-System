#chunk analysis- treba staviti varijabilni chunk size i ako se stavi preveliki da stane kad dode do zadnjeg
                #treba srediti u koji folder idu rezultati kako bi se dalje moglo citati
import pickle

import chunk_analysis as ca
import tempfile_creator as tc
import train_and_validate_creator as tvc
import new_kt as kt
import new_rs as rs


path='/content/Exercise-Recommendation-System/data/biology30/biology30.csv'
chunks=ca.get_chunks(path)
print(chunks[0])

filename='biology30'
path_to_dir='/content/Exercise-Recommendation-System/data/biology30/'

print(len(chunks))
exercise_concepts_mapping,exercises_id_converter=tc.get_mappings(chunks[0])
print(exercise_concepts_mapping, end='\n')
print(exercises_id_converter,end='\n')

no_exercises,no_concepts=ca.get_info(chunks,0)
#TREBA OD NEGDJE DOBITI LISTU SVIH STUDENT TRACEOVA I CANDIDATE EXERCISES za sada i dalje hardkodirati

#kt

tvc.create(filename,path_to_dir)

params= kt.main(filename,path_to_dir)#za sada se train i valid citaju iz csv-a, trebalo bi skuziti kako ih dobiti ko datafreameove iz skripte za stvaranje
#file = open("checkpoint/biology30_32batch_1epochs/kt_params",'rb')
#params = pickle.load(file)


stu=[[(1, 0), (27, 1)]]
cands=[1, 15, 16, 27]
recommendation=rs.run_rs(stu,cands,params,exercise_concepts_mapping,exercises_id_converter,no_exercises,no_concepts)