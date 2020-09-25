import argparse
import os

import tensorflow as tf

from best_data_loader import *
from knowledge_tracing.model import Model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--train', type=str2bool, default='t')
    parser.add_argument('--init_from', type=str2bool, default='t')
    parser.add_argument('--show', type=str2bool, default='f')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--anneal_interval', type=int, default=20)
    parser.add_argument('--maxgradnorm', type=float, default=50.0)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--initial_lr', type=float, default=0.05)
    # synthetic / assist2009_updated / assist2015 / STATICS / biology30
    dataset = 'assist2009_updated'

    # if dataset == 'assist2009_updated':
    #     parser.add_argument('--batch_size', type=int, default=32)
    #     parser.add_argument('--memory_size', type=int, default=9)
    #     parser.add_argument('--memory_key_state_dim', type=int, default=50)
    #     parser.add_argument('--memory_value_state_dim', type=int, default=200)
    #     parser.add_argument('--final_fc_dim', type=int, default=50)
    #     parser.add_argument('--n_questions', type=int, default=2446)
    #     parser.add_argument('--seq_len', type=int, default=200)

    if dataset == 'assist2009_updated':
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--memory_size', type=int, default=10)
        parser.add_argument('--memory_key_state_dim', type=int, default=50)
        parser.add_argument('--memory_value_state_dim', type=int, default=200)
        parser.add_argument('--final_fc_dim', type=int, default=50)
        parser.add_argument('--n_questions', type=int, default=110)
        parser.add_argument('--seq_len', type=int, default=200)

    elif dataset == 'synthetic':
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--memory_size', type=int, default=5)
        parser.add_argument('--memory_key_state_dim', type=int, default=10)
        parser.add_argument('--memory_value_state_dim', type=int, default=10)
        parser.add_argument('--final_fc_dim', type=int, default=50)
        parser.add_argument('--n_questions', type=int, default=50)
        parser.add_argument('--seq_len', type=int, default=50)

    elif dataset == 'assist2015':
        parser.add_argument('--batch_size', type=int, default=50)
        parser.add_argument('--memory_size', type=int, default=20)
        parser.add_argument('--memory_key_state_dim', type=int, default=50)
        parser.add_argument('--memory_value_state_dim', type=int, default=100)
        parser.add_argument('--final_fc_dim', type=int, default=50)
        parser.add_argument('--n_questions', type=int, default=100)
        parser.add_argument('--seq_len', type=int, default=200)

    elif dataset == 'STATICS':
        parser.add_argument('--batch_size', type=int, default=10)
        parser.add_argument('--memory_size', type=int, default=50)
        parser.add_argument('--memory_key_state_dim', type=int, default=50)
        parser.add_argument('--memory_value_state_dim', type=int, default=100)
        parser.add_argument('--final_fc_dim', type=int, default=50)
        parser.add_argument('--n_questions', type=int, default=1223)
        parser.add_argument('--seq_len', type=int, default=200)

    elif dataset == 'a':
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--memory_size', type=int, default=123)
        parser.add_argument('--memory_key_state_dim', type=int, default=50)
        parser.add_argument('--memory_value_state_dim', type=int, default=200)
        parser.add_argument('--final_fc_dim', type=int, default=50)
        parser.add_argument('--n_questions', type=int, default=17751)
        parser.add_argument('--seq_len', type=int, default=200)

    elif dataset == 'biology30':
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--memory_size', type=int, default=5)  # broj koncepata
        parser.add_argument('--memory_key_state_dim', type=int, default=50)
        parser.add_argument('--memory_value_state_dim', type=int, default=200)
        parser.add_argument('--final_fc_dim', type=int, default=50)
        parser.add_argument('--n_questions', type=int, default=30)  # broj pitanja
        parser.add_argument('--seq_len', type=int, default=20)  # prevelik seq_len bacao gresku

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

    data = Data_Loader(args.n_questions, args.seq_len, ',')
    data_directory = os.path.join(args.data_dir, args.dataset)

    with tf.Session(config=run_config) as sess:
        dkvmn = Model(args, sess, name='DKVMN')
        if args.train:
            if dataset == 'synthetic':
                args.dataset = 'naive_c5_q50_s4000_v19'
            train_data_path = os.path.join(data_directory, args.dataset + '_train1.csv')
            valid_data_path = os.path.join(data_directory, args.dataset + '_valid1.csv')

            # train_data_path = 'data/skill_builder/stand_ex_ind_con_ind.csv'
            # valid_data_path = 'data/skill_builder/stand_ex_ind_con_ind.csv'

            train_q_data, train_qa_data = data.load_data(train_data_path)
            print('Train data loaded')
            valid_q_data, valid_qa_data = data.load_data(valid_data_path)
            print('Valid data loaded')
            print('Shape of train data : %s, valid data : %s' % (train_q_data.shape, valid_q_data.shape))
            print('Start training')
            dkvmn.train(train_q_data, train_qa_data, valid_q_data, valid_qa_data)
        # print('Best epoch %d' % (best_epoch))
        else:
            test_data_path = os.path.join(data_directory, args.dataset + '_test.csv')
            test_q_data, test_qa_data = data.load_data(test_data_path)
            print('Test data loaded')
            dkvmn.test(test_q_data, test_qa_data)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'n', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Not expected boolean type')


if __name__ == "__main__":
    main()
