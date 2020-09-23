import tensorflow as tf
from  best_data_loader import *
from  knowledge_tracing.model import Model
import os, argparse


def main(dataset,path=None,fromVariable=False,variableTrain=None,variableValid=None):
    print("ZAPOCEO KT MAIN")




    num_epochs=1
    train='t'
    init_from='t'
    show='f'
    checkpoint_dir='checkpoint'
    log_dir='logs'
    data_dir='data'
    anneal_interval=20
    maxgradnorm=50.0
    momentum=0.9
    initial_lr=0.05
    # synthetic / assist2009_updated / assist2015 / STATICS / biology30

    if dataset == 'assist2009_updated':
        batch_size=32
        memory_size=20
        memory_key_state_dim=50
        memory_value_state_dim=200
        final_fc_dim=50
        n_questions=110
        seq_len=200

    elif dataset == 'synthetic':
        batch_size=32
        memory_size=5
        memory_key_state_dim=10
        memory_value_state_dim=10
        final_fc_dim=50
        n_questions=50
        seq_len=50

    elif dataset == 'assist2015':
        batch_size=50
        memory_size=20
        memory_key_state_dim=50
        memory_value_state_dim=100
        final_fc_dim=50
        n_questions=100
        seq_len=200

    elif dataset == 'STATICS':
        batch_size=10
        memory_size=50
        memory_key_state_dim=50
        memory_value_state_dim=100
        final_fc_dim=50
        n_questions=1223
        seq_len=200

    elif dataset == 'a':
        batch_size=32
        memory_size=123
        memory_key_state_dim=50
        memory_value_state_dim=200
        final_fc_dim=50
        n_questions=17751
        seq_len=200

    elif dataset == 'biology30':
        batch_size=32
        memory_size=5
        memory_key_state_dim=50
        memory_value_state_dim=200
        final_fc_dim=50
        n_questions=30
        seq_len=20


    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
        raise Exception('Need data set')

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    data = Data_Loader(n_questions, seq_len, ',')
    data_directory = os.path.join(data_dir, dataset)

    with tf.Session(config=run_config) as sess:
        dkvmn = Model( memory_size=memory_size,batch_size=batch_size,seq_len=seq_len,n_questions=n_questions,memory_key_state_dim=memory_key_state_dim,
                     memory_value_state_dim=memory_value_state_dim,final_fc_dim=final_fc_dim,momentum=momentum,maxgradnorm=maxgradnorm,
                     show=show,init_from=init_from,checkpoint_dir=checkpoint_dir,log_dir=log_dir,dataset=dataset,inital_lr=initial_lr,
                    anneal_interval=anneal_interval,num_epochs=num_epochs,
         sess=sess,name='DKVMN')
        if train:
            if dataset == 'synthetic':
                dataset = 'naive_c5_q50_s4000_v19'
            if fromVariable:
                print("ZAPOCEO LOADANJE2")
                train_q_data, train_qa_data = data.load_data2(variableTrain)
                valid_q_data, valid_qa_data = data.load_data2(variableValid)
            else:
                train_data_path = os.path.join(path, dataset + '_train1.csv')
                valid_data_path = os.path.join(path, dataset + '_valid1.csv')

                # train_data_path = 'data/skill_builder/stand_ex_ind_con_ind.csv'
                # valid_data_path = 'data/skill_builder/stand_ex_ind_con_ind.csv'

                train_q_data, train_qa_data = data.load_data(train_data_path)
                print('Train data loaded')
                valid_q_data, valid_qa_data = data.load_data(valid_data_path)
                print('Valid data loaded')
                print('Shape of train data : %s, valid data : %s' % (train_q_data.shape, valid_q_data.shape))
                print('Start training')
            dkvmn.train(train_q_data, train_qa_data, valid_q_data, valid_qa_data)
            print("VRACAJU SE PARAMETRI")
            return dkvmn.getParams()
        # print('Best epoch %d' % (best_epoch))
        else:
            test_data_path = os.path.join(data_directory, dataset + '_test.csv')
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
