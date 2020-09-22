#chunk analysis- treba staviti varijabilni chunk size i ako se stavi preveliki da stane kad dode do zadnjeg
                #treba srediti u koji folder idu rezultati kako bi se dalje moglo citati
import pickle

import chunk_analysis as ca
import tempfile_creator as tc
import train_and_validate_creator as tvc
import new_kt as kt
import new_rs as rs
import cand_exer_extractor as cand
import kt_algos_master.train_sakt_bio as bio
import student_traces_extractor as traces
import matplotlib.pyplot as plt


def run_all(path_to_dir,dataset_name,sep='\t'):
   # path='data/biology30/biology30.csv'
   # path2='data/skill_builder/skill_builder_data.csv'
    path=path_to_dir+dataset_name+'.csv'
    chunks=ca.get_chunks(path,sep=sep)
    #print(chunks[0])

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
    filename=''
    tvc.create(filename,path_to_dir)
    '''

    train_variable,valid_variable=tvc.create_from_dataframe(chunks[0],exercises_id_converter,csv=False)

    print("TVC RADI")
    #kt
    params= kt.main(dataset_name,fromVariable=True,variableTrain=train_variable,variableValid=valid_variable)#za sada se train i valid citaju iz csv-a, trebalo bi skuziti kako ih dobiti ko datafreameove iz skripte za stvaranje
   # file = open("data/biology30/kt_params",'rb')
    #params = pickle.load(file)
    print("KT RADI")
    relevancy_matrix=bio.colab_run()
    plt.imshow(relevancy_matrix)
    plt.show()
    treshold=10 #najveci broj kandidata
    personal=cand.PersonalCandidates(relevancy_matrix,cand.max_number_of_exercises,treshold,cand.no_normalization)

    #path_to_gform_traces=''
    #all_student_traces= traces.get_traces_from_gforms(path_to_gform_traces)
    #u stu i cands trebaju ici id-ovi kako su originalno zapisani u datasetu...
    stu=[[(1, 0), (27, 1)]]
    stu_only_ids=[el[0] for el in stu[0] if el[1] == 1] #ako je netko krivo rijesio zadatak on bi i dalje trebao ostati u poolu mogucih zadataka
    cands=personal.get_candidates(stu_only_ids)
    cands.extend([el[0] for el in stu[0]])# zbog rs-a u kandidatima moraju biti i traceovi
    cands=list(set(cands))
    print("SAKT preporuka "+str(cands))
    recommendation=rs.run_rs(stu,cands,params,exercise_concepts_mapping,exercises_id_converter,no_exercises,no_concepts)
