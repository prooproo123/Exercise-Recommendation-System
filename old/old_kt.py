import argparse
import pickle

import tensorflow as tf
from sklearn.model_selection import train_test_split

from old.old_data_loader import *
from old.old_mymodel_concat import Model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=2)
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
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--memory_key_state_dim', type=int, default=100)
    parser.add_argument('--memory_value_state_dim', type=int, default=100)
    parser.add_argument('--final_fc_dim', type=int, default=100)
    parser.add_argument('--seq_len', type=int, default=200)
    parser.add_argument('--seq_iddiff', type=int, default=30)
    parser.add_argument('--count', type=int, default=0)
    dataset = 'skill_builder'
    parser.add_argument('--memory_size', type=int, default=9)
    parser.add_argument('--n_questions', type=int, default=2446)

    args = parser.parse_args()
    args.dataset = dataset

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    # todo
    data = DATA_LOADER(number_AllConcepts=9, number_concepts=1, seqlen=args.seq_len, n_questions=args.n_questions)
    with tf.Session(config=run_config) as sess:
        dkvmn = Model(args, sess, name='DKVMN')

        with open('../data/skill_builder/temp.pkl', 'rb') as f:
            temp = pickle.load(f)

        # exercises ID
        q_datas = [d[0] for d in temp]
        # cross feature of exercise and answer result
        qa_datas = [d[1] for d in temp]
        # knowledge concepts tags
        kgs = [d[2] for d in temp]
        # One Hot Coding Form of Knowledge Concepts
        kg_hot = [d[3] for d in temp]

        if args.train:
            print('Start training')
            for i in range(50):
                q_train, q_valid, qa_train, qa_valid, kg_hot_train, kg_hot_valid, kg_train, kg_valid = train_test_split(
                    q_datas, qa_datas, kg_hot, kgs, test_size=0.3)

                train_q_datas, train_qa_datas, train_kg_datas, train_kgnum_datas = data.load_dataes(
                    q_train,
                    qa_train,
                    kg_hot_train,
                    kg_train,
                )
                valid_q_datas, valid_qa_datas, valid_kg_datas, valid_kgnum_datas = data.load_dataes(
                    q_valid,
                    qa_valid,
                    kg_hot_valid,
                    kg_valid,
                )

                dkvmn.train(train_q_datas, train_qa_datas, valid_q_datas, valid_qa_datas, train_kg_datas,
                            valid_kg_datas,
                            train_kgnum_datas, valid_kgnum_datas)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'n', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Not expected boolean type')


if __name__ == "__main__":
    main()
