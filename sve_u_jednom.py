#chunk analysis- treba staviti varijabilni chunk size i ako se stavi preveliki da stane kad dode do zadnjeg
                #treba srediti u koji folder idu rezultati kako bi se dalje moglo citati

import chunk_analysis as ca
import tempfile_creator as tc
import train_and_validate_creator as tvc

path='C:/Users/Admin/Downloads/Dataset.csv'
chunks=ca.get_chunks(path)
print(chunks[0])

print(len(chunks))
exercise_concepts_mapping,exercises_id_converter=tc.get_mappings(chunks[0])
'''
no_exercises,no_concepts=ca.get_info(chunks,0)
#TREBA OD NEGDJE DOBITI LISTU SVIH STUDENT TRACEOVA I CANDIDATE EXERCISES za sada i dalje hardkodirati

#kt
filename='biology30'
tvc.create(filename,path)
params= kt.main(chunks[0],path)#za sada se train i valid citaju iz csv-a, trebalo bi skuziti kako ih dobiti ko datafreameove iz skripte za stvaranje

stu=[[(1, 0), (27, 1)]]
cands=[1, 15, 16, 27]
recommendation=rs.run_rs(stu,cands,params,exercise_concepts_mapping,exercises_id_converter,no_exercises,no_concepts)
'''