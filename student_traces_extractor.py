import pandas as pd


def get_traces_from_gforms(path_to_file):
    df = pd.read_csv(path_to_file, index_col=False)
    traces=[]

    list_of_gform_columns = [column for column in list(df.columns.values) if '[rezultat]' in column]
    df=df[list_of_gform_columns]
    for index,row in df.iterrows():
        correctness_list = []
        row.apply(lambda cell: correctness_list.append(1 if cell == '1.00 / 1' else 0))
        #print(correctness_list)
        #print(len(correctness_list)
        #print(enumerate(correctness_list))
        traces.append(list (enumerate(correctness_list)))

    #print(traces)
    return traces

#path='C:/Users/Admin/Downloads/GoogleFormsDataset.csv'

#problem je sto u klasicnom obliku dataframea tracevi nisu nuzno poredani kronoloski
def get_traces_from_dataframe(df):

    users= df['user_id'].unique()
    student_traces=[]
    for user in users:
        temp=df[df['user_id']==user]
        student_traces.append(list(zip(temp['problem_id'].tolist(),temp['correct'].tolist())))
    return student_traces
    #vraca traceve za svakog studenta


def convert_and_filter_traces_by_chunk(exercise_id_converter,traces):

    keys=exercise_id_converter.keys()
    new_traces=[]
    for trace in traces:
        if trace[0]  in keys: # ne uzima traceove koji se ne nalaze u chunku
            new_traces.append((exercise_id_converter[trace[0]],trace[1]))
    #treba maknuti iz tracesa zadatke koji se ne nalaze u chunku
    #i pomocu konvertera pretvoriti prave exercise_id-ove u prilagodene

    return new_traces