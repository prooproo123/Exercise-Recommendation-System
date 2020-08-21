import numpy as np
import pandas as pd


class DATA_LOADER():
    def __init__(self, number_AllConcepts, number_concepts, seqlen,n_questions):
        """
        Preprocessing data
        :param number_AllConcepts: the number of all unique knowledge concepts
        :param number_concepts: the number of knowledge concepts for an exercise
        :param seqlen: exercises sequence length
        """
        #123,1, cca 1900
        self.number_AllConcepts = number_AllConcepts
        self.number_concepts = number_concepts
        self.seq_len = seqlen
        self.seperate_char=','
        self.n_questions=n_questions


    def load_data(self, path):
        f_data = open(path, 'r')
        # Question/Answer container
        q_data = list()
        qa_data = list()
        kg_data = list()
        # Read data
        for lineid, line in enumerate(f_data):
            # strip
            line = line.strip()
            # Exercise tag line
            if lineid % 4 == 1:
                # split by ',', returns tag list
                print
                'Excercies tag',
                q_tag_list = line.split(self.seperate_char)

            # Answer
            elif lineid % 4 == 2:
                print(', Answers')
                answer_list = line.split(self.seperate_char)

                # Divide case by seq_len
                if len(q_tag_list) > self.seq_len:
                    n_split = len(q_tag_list) // self.seq_len
                    if len(q_tag_list) % self.seq_len:
                        n_split += 1
                else:
                    n_split = 1
                print('Number of split : %d' % n_split)

            elif lineid % 4 == 3:
                print(', Concepts')
                kg_list = line.split(self.seperate_char)

                # Contain as many as seq_len, then contain remainder
                for k in range(n_split):
                    q_container = list()
                    qa_container = list()
                    kg_container = list()
                    # Less than 'seq_len' element remained
                    if k == n_split - 1:
                        end_index = len(answer_list)
                    else:
                        end_index = (k + 1) * self.seq_len
                    for i in range(k * self.seq_len, end_index):
                        # answers in {0,1}
                        qa_values = int(q_tag_list[i]) + int(answer_list[i]) * self.n_questions
                        q_container.append(int(q_tag_list[i]))
                        qa_container.append(qa_values)
                        kg_container.append(kg_list[i])
                        print('Question tag : %s, Answer : %s, QA : %s' % (q_tag_list[i], answer_list[i], qa_values))
                    # List of list(seq_len, seq_len, seq_len, less than seq_len, seq_len, seq_len...
                    q_data.append(q_container)
                    qa_data.append(qa_container)
                    kg_data.append(kg_container)
        f_data.close()

        # Convert it to numpy array
        q_data_array = np.zeros((len(q_data), self.seq_len))
        for i in range(len(q_data)):
            data = q_data[i]
            # if q_data[i] less than seq_len, remainder would be 0
            q_data_array[i, :len(data)] = data

        qa_data_array = np.zeros((len(qa_data), self.seq_len))
        for i in range(len(qa_data)):
            data = qa_data[i]
            # if qa_data[i] less than seq_len, remainder would be 0
            qa_data_array[i, :len(data)] = data

        kg_data_array = np.zeros((len(kg_data), self.seq_len))
        for i in range(len(kg_data)):
            data = kg_data[i]
            # if kg_data[i] less than seq_len, remainder would be 0
            kg_data_array[i, :len(data)] = data

        return q_data_array,qa_data_array, kg_data_array

    def load_dataes(self, q_data, qa_data, kc_onehots, kcs):

        q_data_array = np.zeros((len(q_data), self.seq_len))
        for i in range(len(q_data)):
            data = q_data[i]
            if len(data) > self.seq_len: continue
            q_data_array[i, :len(data)] = data

        qa_data_array = np.zeros((len(qa_data), self.seq_len))
        for i in range(len(qa_data)):
            data = qa_data[i]
            if len(data) > self.seq_len: continue
            qa_data_array[i, :len(data)] = data

        kg_data_array = np.zeros((len(kc_onehots), self.seq_len, self.number_AllConcepts))
        for i in range(len(kc_onehots)):
            data = np.array(kc_onehots[i])
            if len(data) > self.seq_len: continue
            kg_data_array[i, :len(data)] = data

        kgnum_data_array = np.zeros((len(kcs), self.seq_len,self.number_concepts))
        #kgnum_data_array = np.zeros((len(kcs), self.seq_len, self.number_concepts))
        for i in range(len(kcs)):
            data = np.array(kcs[i])
            if len(data) > self.seq_len: continue
            kgnum_data_array[i, :len(data)] = data

        return q_data_array, qa_data_array, kg_data_array, kgnum_data_array

    def load_dataes2(self, q_data, qa_data, kg_data, kg_num):

        q_data_array = np.zeros((len(q_data), self.seq_len))
        for i in range(len(q_data)):
            data = q_data[i]
            q_data_array[i, :len(data)] = data

        qa_data_array = np.zeros((len(qa_data), self.seq_len))
        for i in range(len(qa_data)):
            data = qa_data[i]
            qa_data_array[i, :len(data)] = data

        kg_data_array = np.zeros((len(kg_data), self.seq_len, self.number_AllConcepts))
        for i in range(len(kg_data)):
            data = np.array(kg_data[i])
            kg_data_array[i, :len(data)] = data

        kgnum_data_array = np.zeros((len(kg_num), self.seq_len, self.number_concepts))
        for i in range(len(kg_num)):
            data = np.array(kg_num[i])
            kgnum_data_array[i, :len(data)] = data

        return q_data_array, qa_data_array, kg_data_array, kgnum_data_array

    def load_dataset2(self, q_data, qa_data, kg_data, kg_num, time, guan, diff):

        q_data_array = np.pad(q_data, self.seq_len)
        qa_data_array = np.pad(qa_data, self.seq_len)
        kg_data_array = np.pad(kg_data, self.seq_len)
        kgnum_data_array = np.pad(kg_num, self.seq_len)
        time_data_array = np.pad(time, self.seq_len)
        guan_data_array = np.pad(guan, self.seq_len)
        diff_data_array = np.pad(diff, self.seq_len)

        # q_data_array=np.pad(pd.Series.to_numpy(q_data),self.seq_len)
        # qa_data_array=np.pad(pd.Series.to_numpy(qa_data),self.seq_len)
        # kg_data_array=np.pad(pd.Series.to_numpy(kg_data),self.seq_len)
        # kgnum_data_array=np.pad(pd.Series.to_numpy(kg_num),self.seq_len)
        # time_data_array=np.pad(pd.Series.to_numpy(time),self.seq_len)
        # guan_data_array=np.pad(pd.Series.to_numpy(guan),self.seq_len)
        # diff_data_array=np.pad(pd.Series.to_numpy(diff),self.seq_len)

        return q_data_array, qa_data_array, kg_data_array, kgnum_data_array, time_data_array, guan_data_array, diff_data_array
