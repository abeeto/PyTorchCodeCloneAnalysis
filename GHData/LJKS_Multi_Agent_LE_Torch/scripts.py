import training
import marl_training
import agents
import commentary_networks
import data
import argparse
import os
import torch
import random
import string
import pickle
from datetime import datetime
from tqdm import tqdm

import utils


def save_hyperparams(path, hyperparams):
    with open(path + 'hyperparameters.pickle', 'wb') as file:
        pickle.dump(hyperparams, file)
    hyperparam_dict = vars(hyperparams)
    hyperparameter_lines = [f'{key} : {hyperparam_dict[key]}' for key in hyperparam_dict.keys()]
    with open(path + 'hyperparameters.txt', 'w') as file:
        file.writelines(hyperparameter_lines)



def string_to_agent(desc_string):
    agent_dict = {}
    agent_dict['sender_lstm128'] = lambda: agents.lstm_sender_agent(feature_size=2049, text_embedding_size=128, vocab_size=2000, lstm_size=128,
                                        lstm_depth=2, feature_embedding_hidden_size=64)
    agent_dict['sender_lstm64'] = lambda: agents.lstm_sender_agent(feature_size=2049, text_embedding_size=64, vocab_size=2000, lstm_size=64, lstm_depth=2, feature_embedding_hidden_size=64)
    agent_dict['receiver_lstm128'] = lambda: agents.lstm_receiver_agent(feature_size=2048, text_embedding_size=128, vocab_size=2000, lstm_size=128,
                                            lstm_depth=2, feature_embedding_hidden_size=64, readout_hidden_size=32)
    agent_dict['receiver_lstm64'] = lambda: agents.lstm_receiver_agent(feature_size=2048, text_embedding_size=64, vocab_size=2000, lstm_size=64, lstm_depth=2, feature_embedding_hidden_size=64, readout_hidden_size=32)

    agent_func = agent_dict[desc_string]
    return agent_func()


def random_desc_string():
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(7))

def summarize_key_args(args, key_args):
    summary=''
    args=vars(args)
    for key in key_args:
        summary = summary+'--'+key+'_'+str(args[key])
    return summary

def get_run_date(run_key, extend):
    if not extend:
        run_date = datetime.now().strftime("%m_%d%H:%M")
        with open(f'{os.path.dirname(os.path.abspath(__file__))}/results/rundocs/{run_key}.pickle', 'wb') as file:
            pickle.dump(run_date, file)
    if extend:
        with open(f'{os.path.dirname(os.path.abspath(__file__))}/results/rundocs/{run_key}.pickle', 'rb') as file:
            run_date = pickle.load(file)
    return run_date

def create_exp_dirs(path):
    save_path = path + 'saves/'
    for which_training in ['pretraining', 'finetune']:
        os.makedirs(save_path + which_training, exist_ok=True)
    os.makedirs(path + 'results', exist_ok=True)

def save_pretrained_models(senders, receivers, path):
    for what_agent, networks in zip(['sender', 'receiver'], [senders, receivers]):
        for num, network in enumerate(networks):
            file_name = what_agent + '_' + str(num) + '.pt'
            network_path = path + 'saves/' + 'pretraining/' + file_name
            torch.save(network.state_dict(), network_path)


def tscl_population_training(args):
    print(f'Sees {torch.cuda.device_count()} CUDA devices')
    key_args = ['num_senders', 'num_receivers', 'num_distractors', 'fifo_size', 'tscl_epsilon', 'sender', 'receiver', 'tscl_sampling', 'entropy_regularization']

    if args.tscl_sampling == 'epsilon_greedy':
        key_args.append('tscl_epsilon')
    elif args.tscl_sampling == 'thompson_sampling':
        key_args.append('tscl_thompson_temp')

    run_date = get_run_date(args.run_key, args.extend)
    path = f'results/{args.experiment}/{summarize_key_args(args, key_args)}/{run_date + args.run_key}/'
    path = os.path.dirname(os.path.abspath(__file__)) + '/' + path

    save_path = path + 'saves/'
    num_distractors = args.num_distractors

    device = torch.device('cuda:0')
    num_senders = args.num_senders
    num_receivers = args.num_receivers
    pretraining_lr = args.pretraining_lr
    receiver_lr = args.finetuning_lr
    sender_lr = args.finetuning_lr
    batch_size = args.batch_size
    test_batch_size = args.test_batch_size
    test_steps = args.test_steps

    pretraining_epochs = args.pretraining_epochs
    finetuning_epochs = args.finetuning_epochs

    repeats_per_epoch = args.repeats_per_epoch

    fifo_size = args.fifo_size

    extend = args.extend
    lr_decay = args.lr_decay
    save_every = args.save_every
    entropy_factor = args.entropy_regularization
    tscl_polyak = args.tscl_polyak
    sampling = {}
    sampling['style'] = args.tscl_sampling
    sampling['control'] = args.tscl_thompson_temp if args.tscl_sampling == 'thompson_sampling' else args.tscl_epsilon

    senders = [string_to_agent(args.sender) for _ in range(num_senders)]
    receivers = [string_to_agent(args.receiver) for
                 _ in range(num_receivers)]

    if not args.extend:
        create_exp_dirs(path)
        save_hyperparams(path, args)

        pretrain_sender = lambda sender: training.pretrain_sender_lstm(sender=sender, path=path + 'sender_pretraining', batch_size=batch_size,
                                                                       num_distractors=num_distractors,
                                                                       num_episodes=pretraining_epochs, lr=pretraining_lr,
                                                                       device=device)
        print('Pretraining senders:')
        senders = [pretrain_sender(sender) for sender in tqdm(senders)]

        pretrain_receiver = lambda receiver: training.pretrain_receiver_lstm(receiver=receiver,
                                                                             num_episodes=pretraining_epochs,
                                                                             path=path + 'receiver_pretraining',
                                                                             batch_size=batch_size,
                                                                             num_distractors=num_distractors,
                                                                             lr=pretraining_lr, device=device)
        print('Pretraining receivers:')
        receivers = [pretrain_receiver(receiver) for receiver in tqdm(receivers)]
        save_pretrained_models(senders, receivers, path)


    print('Interactive finetuning')
    marl_training.tscl_multiagent_training_interactive_only(senders=senders, receivers=receivers, receiver_lr=receiver_lr, sender_lr=sender_lr, num_distractors=num_distractors, path=path, sampling=sampling, fifo_size=fifo_size, tscl_polyak=tscl_polyak, num_episodes=finetuning_epochs, batch_size=batch_size, repeats_per_epoch=repeats_per_epoch, device=device, lr_decay=lr_decay, entropy_factor=entropy_factor, save_every=save_every, load_params=extend)


def commentary_idx_training(args):
    print(f'Sees {torch.cuda.device_count()} CUDA devices')
    key_args = ['num_senders', 'num_receivers', 'num_distractors', 'sender', 'receiver']
    path = f'results/{args.experiment}/{summarize_key_args(args, key_args)}/{datetime.now().strftime("%m_%d_%Y,%H:%M:%S")+random_desc_string()}/'
    path = os.path.dirname(os.path.abspath(__file__)) + '/' + path

    save_path = path + 'saves/'
    for which_training in ['pretraining', 'finetuning']:
        for agent in ['sender', 'receiver']:
            sub_dir = which_training + '_' + agent
            os.makedirs(save_path + sub_dir, exist_ok=True)

    save_hyperparams(path, args)

    writer_tag = args.tag
    num_distractors = args.num_distractors

    device = torch.device('cuda:0')
    num_senders = args.num_senders
    num_receivers = args.num_receivers

    pretraining_lr = args.pretraining_lr
    receiver_lr = args.finetuning_lr
    sender_lr = args.finetuning_lr
    commentary_lr = args.commentary_lr

    batch_size = args.batch_size

    pretraining_epochs = args.pretraining_epochs
    finetuning_epochs = args.finetuning_epochs

    repeats_per_epoch = args.repeats_per_epoch

    inner_loop_steps = args.inner_loop_steps
    commentary_nn = commentary_networks.idx_commentary_network(num_senders, num_receivers, 16, 16)

    senders = [string_to_agent(args.sender) for _ in range(num_senders)]
    pretrain_sender = lambda sender: training.pretrain_sender_lstm(sender=sender, path=path + 'sender_pretraining',
                                                                   writer_tag=writer_tag, batch_size=batch_size,
                                                                   num_distractors=num_distractors,
                                                                   num_episodes=pretraining_epochs, lr=pretraining_lr,
                                                                   device=device)
    print('Pretraining senders:')
    senders = [pretrain_sender(sender) for sender in tqdm(senders)]

    receivers = [string_to_agent(args.receiver) for _ in range(num_receivers)]

    pretrain_receiver = lambda receiver: training.pretrain_receiver_lstm(receiver=receiver,
                                                                         num_episodes=pretraining_epochs,
                                                                         path=path + 'receiver_pretraining',
                                                                         writer_tag=writer_tag,
                                                                         batch_size=batch_size,
                                                                         num_distractors=num_distractors,
                                                                         lr=pretraining_lr, device=device)
    print('Pretraining receivers:')
    receivers = [pretrain_receiver(receiver) for receiver in tqdm(receivers)]
    for what_agent, networks in zip(['sender', 'receiver'], [senders, receivers]):
        for num, network in enumerate(networks):
            sub_dir = 'pretraining' + '_' + what_agent + '/'
            file_name = what_agent + '_' + str(num) + '.pt'
            network_path = save_path + sub_dir + file_name
            torch.save(network.state_dict(), network_path)

    marl_training.idx_commentary_training_interactive_only(senders=senders, receivers=receivers,
                                                                        commentary_network=commentary_nn,
                                                                        receiver_lr=receiver_lr, sender_lr=sender_lr,
                                                                        commentary_lr=commentary_lr,
                                                                        num_distractors=num_distractors, path=path,
                                                                        writer_tag=writer_tag,
                                                                        num_inner_loop_steps=inner_loop_steps,
                                                                        num_episodes=finetuning_epochs,
                                                                        batch_size=batch_size, repeats_per_epoch=repeats_per_epoch, device=device)

    for what_agent, networks in zip(['sender', 'receiver'], [senders, receivers]):
        for num, network in enumerate(networks):
            sub_dir = 'finetuning' + '_' + what_agent + '/'
            file_name = what_agent + '_' + str(num) + '.pt'
            network_path = save_path + sub_dir + file_name
            torch.save(network.state_dict(), network_path)

    # save commentary_network
    c_network_path = save_path + 'commentary_network/commment.pt'
    os.makedirs(save_path + 'commentary_network', exist_ok=True)
    torch.save(commentary_nn.state_dict(), c_network_path)

def commentary_weighting_training(args):
    print(f'Sees {torch.cuda.device_count()} CUDA devices')

    key_args = ['num_senders', 'num_receivers', 'num_distractors', 'finetuning_lr', 'batch_size', 'sender', 'receiver']
    path = f'results/{args.experiment}/{summarize_key_args(args, key_args)}/{datetime.now().strftime("%m_%d_%Y,%H:%M:%S")+random_desc_string()}/'
    path = os.path.dirname(os.path.abspath(__file__)) + '/' + path

    save_path = path + 'saves/'
    for which_training in ['pretraining', 'finetuning']:
        for agent in ['sender', 'receiver']:
            sub_dir = which_training + '_' + agent
            os.makedirs(save_path + sub_dir)

    save_hyperparams(path, args)

    writer_tag = args.tag
    num_distractors = args.num_distractors

    device = torch.device('cuda:0')
    num_senders = args.num_senders
    num_receivers = args.num_receivers

    pretraining_lr = args.pretraining_lr
    receiver_lr = args.finetuning_lr
    sender_lr = args.finetuning_lr
    commentary_lr = args.commentary_lr

    batch_size = args.batch_size

    pretraining_epochs = args.pretraining_epochs
    finetuning_epochs = args.finetuning_epochs

    repeats_per_epoch = args.repeats_per_epoch

    inner_loop_steps = args.inner_loop_steps
    commentary_nn = commentary_networks.objects_commentary_network_normalized(num_senders, num_receivers, 64, 2049, 2,
                                                                              2, 64, 4)

    senders = [string_to_agent(args.sender) for _ in range(num_senders)]
    pretrain_sender = lambda sender: training.pretrain_sender_lstm(sender=sender, path=path + 'sender_pretraining',
                                                                   writer_tag=writer_tag, batch_size=batch_size,
                                                                   num_distractors=num_distractors,
                                                                   num_episodes=pretraining_epochs, lr=pretraining_lr,
                                                                   device=device)
    print('Pretraining senders:')
    senders = [pretrain_sender(sender) for sender in tqdm(senders)]

    receivers = [string_to_agent(args.receiver) for _ in range(num_receivers)]

    pretrain_receiver = lambda receiver: training.pretrain_receiver_lstm(receiver=receiver,
                                                                         num_episodes=pretraining_epochs,
                                                                         path=path + 'receiver_pretraining',
                                                                         writer_tag=writer_tag,
                                                                         batch_size=batch_size,
                                                                         num_distractors=num_distractors,
                                                                         lr=pretraining_lr, device=device)
    print('Pretraining receivers:')
    receivers = [pretrain_receiver(receiver) for receiver in tqdm(receivers)]
    for what_agent, networks in zip(['sender', 'receiver'], [senders, receivers]):
        for num, network in enumerate(networks):
            sub_dir = 'pretraining' + '_' + what_agent + '/'
            file_name = what_agent + '_' + str(num) + '.pt'
            network_path = save_path + sub_dir + file_name
            torch.save(network.state_dict(), network_path)

    marl_training.weighted_softmax_commentary_training_interactive_only(senders=senders, receivers=receivers, commentary_network=commentary_nn, receiver_lr=receiver_lr, sender_lr=sender_lr,
                                                          commentary_lr=commentary_lr, num_distractors=num_distractors, path=path, writer_tag=writer_tag, num_inner_loop_steps=inner_loop_steps, num_episodes=finetuning_epochs,
                                                          batch_size=batch_size, repeats_per_epoch=repeats_per_epoch, device=device)

    for what_agent, networks in zip(['sender', 'receiver'], [senders, receivers]):
        for num, network in enumerate(networks):
            sub_dir = 'finetuning' + '_' + what_agent + '/'
            file_name = what_agent + '_' + str(num) + '.pt'
            network_path = save_path + sub_dir + file_name
            torch.save(network.state_dict(), network_path)

    #save commentary_network
    c_network_path = save_path + 'commentary_network/commment.pt'
    os.makedirs(save_path + 'commentary_network')
    torch.save(commentary_nn.state_dict(), c_network_path)


def baseline_population_training(args):
    print(f'Sees {torch.cuda.device_count()} CUDA devices')

    key_args = ['num_senders', 'num_receivers', 'num_distractors', 'finetuning_lr', 'batch_size', 'sender', 'receiver']

    run_date = get_run_date(args.run_key, args.extend)

    path = f'results/{args.experiment}/{summarize_key_args(args, key_args)}/{run_date+args.run_key}/'
    path = os.path.dirname(os.path.abspath(__file__)) + '/' + path
    num_distractors = args.num_distractors

    device = torch.device('cuda:0')
    num_senders = args.num_senders
    num_receivers = args.num_receivers

    pretraining_lr = args.pretraining_lr
    receiver_lr = args.finetuning_lr
    sender_lr = args.finetuning_lr

    batch_size = args.batch_size

    pretraining_epochs = args.pretraining_epochs
    finetuning_epochs = args.finetuning_epochs
    repeats_per_epoch = args.repeats_per_epoch
    extend = args.extend
    lr_decay = args.lr_decay
    save_every = args.save_every
    entropy_factor = args.entropy_regularization
    test_batch_size = args.test_batch_size
    test_steps = args.test_steps

    senders = [string_to_agent(args.sender) for _ in range(num_senders)]
    receivers = [string_to_agent(args.receiver) for _ in range(num_receivers)]

    if not args.extend:
        create_exp_dirs(path)

        save_hyperparams(path, args)

        pretrain_sender = lambda sender: training.pretrain_sender_lstm(sender=sender, path=path + 'sender_pretraining', batch_size=batch_size, num_distractors=num_distractors,
                                      num_episodes=pretraining_epochs, lr=pretraining_lr, device=device)
        print('Pretraining senders:')
        senders = [pretrain_sender(sender) for sender in tqdm(senders)]


        pretrain_receiver = lambda receiver: training.pretrain_receiver_lstm(receiver=receiver, num_episodes=pretraining_epochs,
                                          path=path + 'receiver_pretraining', batch_size=batch_size, num_distractors=num_distractors, lr=pretraining_lr, device=device)
        print('Pretraining receivers:')
        receivers = [pretrain_receiver(receiver) for receiver in tqdm(receivers)]
        save_pretrained_models(senders, receivers, path)




    print('Interactive finetuning')
    marl_training.baseline_multiagent_training_interactive_only(senders, receivers, receiver_lr, sender_lr, num_distractors, path, num_episodes=finetuning_epochs, batch_size=batch_size, repeats_per_epoch=repeats_per_epoch, device=device, baseline_polyak=0.99, lr_decay=lr_decay, entropy_factor=entropy_factor, save_every=save_every, test_batch_size=test_batch_size, test_steps=test_steps, load_params=extend)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='scripts running MARL LE experiments')
    parser.add_argument('--experiment')
    parser.add_argument('--sender', type=str, default='sender_lstm64')
    parser.add_argument('--receiver', type=str, default='receiver_lstm64')
    parser.add_argument('--pretraining_lr', type=float, default=0.0001)
    parser.add_argument('--finetuning_lr', type=float, default=0.00001)
    parser.add_argument('--num_senders', type=int, default=2)
    parser.add_argument('--num_receivers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=8192)
    parser.add_argument('--test_steps', type=int, default=3)
    parser.add_argument('--pretraining_epochs', type=int, default=25)
    parser.add_argument('--finetuning_epochs', type=int, default=200)
    parser.add_argument('--num_distractors', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--fifo_size', type=int, default=10)
    parser.add_argument('--tscl_polyak', type=float, default=0.0)
    parser.add_argument('--tscl_epsilon', type=float, default=0.1)
    parser.add_argument('--tscl_thompson_temp', type=float, default=100)
    parser.add_argument('--commentary_lr', type=float, default=0.00001)
    parser.add_argument('--inner_loop_steps', type=int, default=2)
    parser.add_argument('--repeats_per_epoch', type=int, default=1)
    parser.add_argument('--entropy_regularization', type=float, default=0.0)
    parser.add_argument('--run_key', type=str, default='default', required=True)
    parser.add_argument('--extend', type=str, default='False', choices=['True', 'False'])
    parser.add_argument('--lr_decay', type=float, default=1.0)
    parser.add_argument('--save_every', type=int, default=25)
    parser.add_argument('--forceload_args', type=bool, default=False)
    parser.add_argument('--tscl_sampling', type=str, default='epsilon_greedy', choices=['epsilon_greedy','thompson_sampling'])

    script_dict = {'baseline_population_training':baseline_population_training, 'tscl_population_training':tscl_population_training, 'commentary_weighting_training':commentary_weighting_training, 'commentary_idx_training':commentary_idx_training}
    args = parser.parse_args()

    #Input string as boolean is broken in Python, have to convert manually!
    args.extend = utils.util_bool_string(args.extend)
    print(vars(args))
    #make sure the run is legit, i.e. if a run is to be extended, the run does already exist, if start from scratch make sure a new key is used!
    os.makedirs(os.path.dirname(os.path.abspath(__file__)) + '/results/rundocs', exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(__file__)) + '/results/rundocs/runargs', exist_ok=True)

    if args.extend:
        assert os.path.isfile(f'{os.path.dirname(os.path.abspath(__file__))}/results/rundocs/{args.run_key}.pickle')
        if args.forceload_args:
            with open(f'{os.path.dirname(os.path.abspath(__file__))}/results/rundocs/runargs/{args.run_key}.pickle', 'rb') as file:
                args = pickle.load(file)
                args.extend = True
    else:
        assert not os.path.isfile(f'{os.path.dirname(os.path.abspath(__file__))}/results/rundocs/{args.run_key}.pickle')
        with open(f'{os.path.dirname(os.path.abspath(__file__))}/results/rundocs/runargs/{args.run_key}.pickle', 'wb') as file:
            pickle.dump(args, file)
    #run the script!
    script_dict[args.experiment](args)

