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

