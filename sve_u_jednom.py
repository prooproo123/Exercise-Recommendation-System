#chunk analysis- treba staviti varijabilni chunk size i ako se stavi preveliki da stane kad dode do zadnjeg
                #treba srediti u koji folder idu rezultati kako bi se dalje moglo citati
import pickle

import chunk_analysis as ca
import tempfile_creator as tc
import train_and_validate_creator as tvc
import new_kt as kt
import new_rs as rs

chunk_filename='chunk.csv'
path='/content/gdrive/My Drive/data/skill_builder_data.csv'
chunks=ca.get_chunks(path)
#print(chunks[3])

filename='assist2009_updated'
path_to_dir='/content/gdrive/My Drive/data/'

print(len(chunks))
exercise_concepts_mapping,exercises_id_converter,exercises=tc.get_mappings(chunks[3])
#exercise_concepts_mapping,exercises_id_converter=tc.get_mappings(path_to_dir,chunk_filename)
print(exercise_concepts_mapping, end='\n')
print(exercises_id_converter,end='\n')
#print(exercises,end='\n')

no_exercises,no_concepts,chunk_filename=ca.get_info(path_to_dir,chunks,3)
#TREBA OD NEGDJE DOBITI LISTU SVIH STUDENT TRACEOVA I CANDIDATE EXERCISES za sada i dalje hardkodirati

#kt

tvc.create(chunk_filename,filename,path_to_dir)

params= kt.main(filename,path_to_dir)#za sada se train i valid citaju iz csv-a, trebalo bi skuziti kako ih dobiti ko datafreameove iz skripte za stvaranje
#file = open("checkpoint/biology30_32batch_1epochs/kt_params",'rb')
#params = pickle.load(file)


stu = [[(85829, 0),(85838, 1)]]
cands=[85829,61089,85814,85838]
recommendation=rs.run_rs(stu,cands,params,exercise_concepts_mapping,exercises_id_converter,no_exercises,no_concepts)