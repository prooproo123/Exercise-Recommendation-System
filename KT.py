import argparse
import os
import pickle

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from data_loader import *
from mymodel_concat import Model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--train', type=str2bool, default='t')
    parser.add_argument('--init_from', type=str2bool, default='t')
    parser.add_argument('--show', type=str2bool, default='f')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--data_dir', type=str, default='/home/zvonimir/Exercise-Recommendation-System/data/')
    parser.add_argument('--anneal_interval', type=int, default=20)
    parser.add_argument('--maxgradnorm', type=float, default=50.0)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--initial_lr', type=float, default=0.05)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--memory_key_state_dim', type=int, default=100)
    parser.add_argument('--memory_value_state_dim', type=int, default=100)
    parser.add_argument('--final_fc_dim', type=int, default=100)
    parser.add_argument('--seq_len', type=int, default=150)
    parser.add_argument('--seq_iddiff', type=int, default=30)
    parser.add_argument('--count', type=int, default=0)
    dataset = 'skill_builder_data.csv'
    parser.add_argument('--memory_size', type=int, default=188)
    parser.add_argument('--n_questions', type=int, default=1982)
    print(dataset)

    args = parser.parse_args()
    args.dataset = dataset

    print(args)
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    if not os.path.exists(args.data_dir):
        os.mkdir(args.data_dir)
        raise Exception('Need data set')

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    data = DATA_LOADER(args.memory_size, 3, args.seq_len)
    data_directory = os.path.join(args.data_dir, args.dataset)

    # history_path = '/home/zvonimir/Exercise-Recommendation-System/data/skill_builder_data.csv'
    #
    # dffff = pd.read_csv(history_path,
    #                  dtype={'order_id': int, 'assignment_id': int, 'user_id': int, 'assistment_id': int,
    #                         'problem_id': int,
    #                         'original': int, 'correct': int, 'attempt_count': int, 'ms_first_response': int,
    #                         'tutor_mode': 'string', 'answer_type': 'string', 'sequence_id': int,
    #                         'student_class_id': int,
    #                         'position': int, 'type': 'string', 'base_sequence_id': int, 'skill_id': float,
    #                         'skill_name': 'string',
    #                         'teacher_id': int, 'school_id': int, 'hint_count': int, 'hint_total': int,
    #                         'overlap_time': int,
    #                         'template_id': int, 'answer_id': int, 'answer_text': 'string', 'first_action': int,
    #                         'bottom_hint': int, 'opportunity': int, 'opportunity_original': int
    #                         },
    #                  usecols=['order_id', 'assignment_id', 'user_id', 'assistment_id', 'problem_id', 'original',
    #                           'correct',
    #                           'attempt_count', 'ms_first_response', 'tutor_mode', 'answer_type', 'sequence_id',
    #                           'student_class_id', 'position', 'type', 'base_sequence_id', 'skill_id', 'skill_name',
    #                           'teacher_id', 'school_id', 'hint_count', 'hint_total', 'overlap_time', 'template_id',
    #                           'first_action', 'opportunity', ])


    #pkl_path = '/home/zvonimir/Exercise-Recommendation-System/data/skill_builder_pickle.pkl'

    # with open(pkl_path, 'wb') as f:
    #     pickle.dump(dffff, f, pickle.HIGHEST_PROTOCOL)

    with tf.Session(config=run_config) as sess:
        dkvmn = Model(args, sess, name='DKVMN')
        # dkvmn.getParam()
        with open(pkl_path, 'rb') as f:
            df = pickle.load(f)

        #kg_hot = df['skill_id']  # onehot
        #knowledge concepts==skills
        knowledge_concepts_onehots = pd.get_dummies(df['skill_id']).values

        knowledge_concepts_ids = df['skill_id'].values
        problem_ids = df['problem_id'].values



        q=qa_datas = df['problem_id'].values
        a=qa_datas = df['correct'].values
        qa_datas = df['skill_id'].values



        diff = df['ms_first_response'].values/100



        response_time = df['ms_first_response'].values


        guan = df['ms_first_response'].values/1000



    # # One Hot Coding Form of Knowledge Concepts
    # kg_hot = [d[0] for d in temp]
    # # knowledge concepts tags
    # kgs = [d[1] for d in temp]
    # # exercises ID
    # q_datas = [d[2] for d in temp]
    # # cross feature of exercise and answer result
    # qa_datas = [d[3] for d in temp]
    # # difficulty of exercises
    # diff = [d[4] for d in temp]
    # # Exercise completion time
    # dotime = [d[6] for d in temp]
    # # the gate of exercises
    # guan = [d[5] for d in temp]

        if args.train:
            print('Start training')
            for i in range(50):
                q_train, q_valid, qa_train, qa_valid, kg_hot_train, kg_hot_valid, kg_train, kg_valid, traintime, \
                validtime, trainguan, validguan, traindiff, validdiff = train_test_split(
                    problem_ids, qa_datas, knowledge_concepts_onehots, knowledge_concepts_ids, response_time, guan, diff, test_size=0.3)

                train_q_datas, train_qa_datas, train_kg_datas, train_kgnum_datas, train_time, train_guan, train_diff = data.load_dataset2(
                    q_train,
                    qa_train,
                    kg_hot_train,
                    kg_train,
                    traintime,
                    trainguan,
                    traindiff)
                # train_q_datas, train_qa_datas, train_kg_datas, train_kgnum_datas, train_time, train_guan, train_diff = data.load_dataes(
                #     q_train,
                #     qa_train,
                #     kg_hot_train,
                #     kg_train,
                #     traintime,
                #     trainguan,
                #     traindiff)
                # valid_q_datas, valid_qa_datas, valid_kg_datas, valid_kgnum_datas, valid_time, valid_guan, valid_diff = data.load_dataes(
                #     q_valid,
                #     qa_valid,
                #     kg_hot_valid,
                #     kg_valid,
                #     validtime,
                #     validguan,
                #     validdiff)
                valid_q_datas, valid_qa_datas, valid_kg_datas, valid_kgnum_datas, valid_time, valid_guan, valid_diff = data.load_dataset2(
                    q_valid,
                    qa_valid,
                    kg_hot_valid,
                    kg_valid,
                    validtime,
                    validguan,
                    validdiff)

                dkvmn.train(train_q_datas, train_qa_datas, valid_q_datas, valid_qa_datas, train_kg_datas,
                            valid_kg_datas,
                            train_kgnum_datas, valid_kgnum_datas, train_time, valid_time, train_guan, valid_guan,
                            train_diff, valid_diff)


# print('Best epoch %d' % (best_epoch))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'n', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Not expected boolean type')


if __name__ == "__main__":
    main()
