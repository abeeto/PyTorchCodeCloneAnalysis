import torch
import data
import utils
import agents
import torch.nn.functional as F
import torchmetrics
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

TEST_VAL_SPLIT = 0.5 #of the designated 'test' data we treat the first 'TEST_VAL_SPLIT' as test data by default, the rest as validation






def target_distractor_encode_data(features_batch, target_idx_batch, num_classes):
    onehot_encoding = F.one_hot(target_idx_batch, num_classes=num_classes)
    onehot_encoding = torch.swapaxes(onehot_encoding, 1,2) # (batchsize, 1, numtargets+distracts) TO (batchsize, numtargets+distracts, 1)
    target_encoded_features = torch.cat((onehot_encoding, features_batch), dim=2)
    return  target_encoded_features


def prob_mask(tokens, eos_token=4):
    #only include timesteps before and including endofsequence
    #creates a mask that is 0 for all elements after the eostoken has been reached, 1 else

    #check where you find eos tokens:
    eos_tokens = tokens==eos_token

    #mark everything that is or is after a eos token
    eos_tokens = torch.cumsum(eos_tokens, dim=1)
    #do it twice so the first eos token has a value of 1, all later tokens are > 1
    eos_tokens = torch.cumsum(eos_tokens, dim=1)


    #now we have counts but we want whether count is <=1 as a float (to include the eos token!)

    eos_tokens = (eos_tokens<=1.).to(dtype=float)

    return eos_tokens

def pretrain_sender_transfomer(sender, num_episodes, path, batch_size=128, num_distractors=7, lr=0.0001, num_workers=4, device='cpu'):
    sender.to(device)
    sender.train()
    optimizer = torch.optim.Adam(sender.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    (train_ds, test_ds), vocab = data.load_prepared_coco_data()

    data_loader_train = data.create_data_loader(
        train_ds, batch_size=batch_size, num_distractors=num_distractors, num_workers=num_workers, device=device
    )
    #split test data in val and test data
    test_val_cutoff = int(len(train_ds)*TEST_VAL_SPLIT)
    validation_ds = test_ds[test_val_cutoff:]
    test_ds = test_ds[:test_val_cutoff]
    data_loader_test = data.create_data_loader(
        test_ds, batch_size=batch_size, num_distractors=num_distractors, num_workers=num_workers, device=device
    )
    #metrics
    writer = SummaryWriter(path)
    training_loss = torchmetrics.MeanMetric()
    training_pp = torchmetrics.MeanMetric()
    test_loss = torchmetrics.MeanMetric()
    test_pp = torchmetrics.MeanMetric()

    vocab_size = len(vocab['idx2word'])

    for episode in (p_bar := tqdm(range(num_episodes))):
        sender.train()

        for all_features_batch, target_features_batch, target_captions_stack, target_idx_batch, ids_batch in data_loader_train:
            optimizer.zero_grad()
            all_features_batch = all_features_batch.to(device=device)
            target_captions_stack = target_captions_stack.to(device=device)
            target_idx_batch = target_idx_batch.to(device=device)
            target_encoded_features = target_distractor_encode_data(all_features_batch, target_idx_batch, num_distractors+1)
            logits = sender(target_encoded_features, target_captions_stack[:, :-1], device=device)
            loss = criterion(torch.swapaxes(logits, 1,2), target_captions_stack[:,1:])
            loss = loss*prob_mask(target_captions_stack)[:,1:]
            loss = torch.mean(loss)
            loss.backward()
            optimizer.step()
            #print(loss.to('cpu'))

            pp = torch.exp(loss)
            training_loss.update(loss.to('cpu'))
            training_pp.update(pp.to('cpu'))

        #test
        sender.eval()
        #write one batch of test elems as sentences
        write_batch = True
        write_string = None
        for all_features_batch, target_features_batch, target_captions_stack, target_idx_batch, ids_batch in data_loader_test:
            all_features_batch = all_features_batch.to(device=device)
            target_captions_stack = target_captions_stack.to(device=device)
            target_idx_batch = target_idx_batch.to(device=device)
            target_encoded_features = target_distractor_encode_data(all_features_batch, target_idx_batch,
                                                                    num_distractors + 1)
            logits = sender(target_encoded_features, target_captions_stack[:, :-1], device=device)
            if write_batch:
                argmax_seqs = torch.argmax(logits, dim=-1).cpu().detach()
                test_sentences = utils.batch2sentences(vocab, argmax_seqs)
                org_sentences = utils.batch2sentences(vocab, target_captions_stack.cpu().detach())
                write_string = utils.summarize_test_sentences(org_sentences, test_sentences)
                write_batch = False

            loss = criterion(torch.swapaxes(logits, 1, 2), target_captions_stack[:, 1:])
            loss = loss*prob_mask(target_captions_stack)[:,1:]
            loss = torch.mean(loss)
            pp = torch.exp(loss)
            test_loss.update(loss.to('cpu'))
            test_pp.update(pp.to('cpu'))

        scheduler.step()
        episode_training_loss = training_loss.compute()
        training_loss.reset()
        episode_training_pp = training_pp.compute()
        training_pp.reset()
        episode_test_loss = test_loss.compute()
        test_loss.reset()
        episode_test_pp = test_pp.compute()
        test_pp.reset()

        p_bar.set_description(f'Train: L{episode_training_loss :.3e} / PP{episode_training_pp :.3e} || Test: L{episode_test_loss :.3e} / PP{episode_test_pp :.3e}')
        writer.add_scalars(main_tag='todotag', tag_scalar_dict={'training_loss': episode_training_loss, 'training_pp': episode_training_pp, 'test_loss': episode_test_loss, 'test_pp': episode_test_pp}, global_step=episode)
        writer.add_text(tag='test_sentences', text_string=write_string, global_step=episode)

    writer.flush()
    return sender

def pretrain_sender_lstm(sender, num_episodes, path,  batch_size=128, num_distractors=7, lr=0.0001, num_workers=4, device='cpu'):
    sender.to(device)
    sender.train()
    optimizer = torch.optim.Adam(sender.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    (train_ds, test_ds), vocab = data.load_prepared_coco_data()

    data_loader_train = data.create_data_loader(
        train_ds, batch_size=batch_size, num_distractors=num_distractors, num_workers=num_workers, device=device
    )
    #split test data in val and test data
    test_val_cutoff = int(len(train_ds)*TEST_VAL_SPLIT)
    validation_ds = test_ds[test_val_cutoff:]
    test_ds = test_ds[:test_val_cutoff]
    data_loader_test = data.create_data_loader(
        test_ds, batch_size=batch_size, num_distractors=num_distractors, num_workers=num_workers, device=device
    )
    #metrics
    writer = SummaryWriter(path)
    training_loss = torchmetrics.MeanMetric()
    training_pp = torchmetrics.MeanMetric()
    test_loss = torchmetrics.MeanMetric()
    test_pp = torchmetrics.MeanMetric()

    vocab_size = len(vocab['idx2word'])

    for episode in (p_bar := tqdm(range(num_episodes))):
        sender.train()
        for all_features_batch, target_features_batch, target_captions_stack, target_idx_batch, ids_batch in data_loader_train:
            optimizer.zero_grad()
            all_features_batch = all_features_batch.to(device=device)
            target_captions_stack = target_captions_stack.to(device=device)
            target_idx_batch = target_idx_batch.to(device=device)
            target_encoded_features = target_distractor_encode_data(all_features_batch, target_idx_batch, num_distractors+1)
            logits = sender(target_encoded_features, target_captions_stack[:, :-1], device=device)
            loss = criterion(torch.swapaxes(logits, 1,2), target_captions_stack[:,1:])
            #print(loss.size(), prob_mask(prob_mask(target_captions_stack))[:,1:].size())
            #print(prob_mask(target_captions_stack)[:,1:].cpu().detach())
            loss = loss*prob_mask(target_captions_stack)[:,1:]
            loss = torch.mean(loss)
            loss.backward()
            optimizer.step()
            #print(loss.to('cpu'))

            pp = torch.exp(loss)
            training_loss.update(loss.to('cpu'))
            training_pp.update(pp.to('cpu'))

        #test
        sender.eval()
        #write one batch of test elems as sentences
        write_batch = True
        write_string = None

        for all_features_batch, target_features_batch, target_captions_stack, target_idx_batch, ids_batch in data_loader_test:
            all_features_batch = all_features_batch.to(device=device)
            target_captions_stack = target_captions_stack.to(device=device)
            target_idx_batch = target_idx_batch.to(device=device)
            target_encoded_features = target_distractor_encode_data(all_features_batch, target_idx_batch,
                                                                    num_distractors + 1)
            logits = sender(target_encoded_features, target_captions_stack[:, :-1], device=device)

            if write_batch:
                argmax_seqs = torch.argmax(logits, dim=-1).cpu().detach()
                test_sentences = utils.batch2sentences(vocab, argmax_seqs)
                org_sentences = utils.batch2sentences(vocab, target_captions_stack.cpu().detach())
                write_string = utils.summarize_test_sentences(org_sentences, test_sentences)
                write_batch = False
            loss = criterion(torch.swapaxes(logits, 1, 2), target_captions_stack[:, 1:])
            loss = loss*prob_mask(target_captions_stack)[:,1:]
            loss = torch.mean(loss)
            pp = torch.exp(loss)
            test_loss.update(loss.to('cpu'))
            test_pp.update(pp.to('cpu'))

        scheduler.step()
        episode_training_loss = training_loss.compute()
        training_loss.reset()
        episode_training_pp = training_pp.compute()
        training_pp.reset()
        episode_test_loss = test_loss.compute()
        test_loss.reset()
        episode_test_pp = test_pp.compute()
        test_pp.reset()

        p_bar.set_description(f'Train: L{episode_training_loss :.3e} / PP{episode_training_pp :.3e} || Test: L{episode_test_loss :.3e} / PP{episode_test_pp :.3e}')
        writer.add_scalars(main_tag='todotag', tag_scalar_dict={'training_loss': episode_training_loss, 'training_pp': episode_training_pp, 'test_loss': episode_test_loss, 'test_pp': episode_test_pp}, global_step=episode)
        writer.add_text(tag='test_sentences', text_string=write_string, global_step=episode)
    writer.flush()
    return sender


def pretrain_receiver_transformer(receiver, num_episodes, path,  batch_size=128, num_distractors=7, lr=0.0001, num_workers=4, device='cpu'):
    receiver.to(device)
    receiver.train()
    optimizer = torch.optim.Adam(receiver.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    criterion = torch.nn.CrossEntropyLoss()

    (train_ds, test_ds), vocab = data.load_prepared_coco_data()

    data_loader_train = data.create_data_loader(
        train_ds, batch_size=batch_size, num_distractors=num_distractors, num_workers=num_workers, device=device
    )
    #split test data in val and test data
    test_val_cutoff = int(len(train_ds)*TEST_VAL_SPLIT)
    validation_ds = test_ds[test_val_cutoff:]
    test_ds = test_ds[:test_val_cutoff]
    data_loader_test = data.create_data_loader(
        test_ds, batch_size=batch_size, num_distractors=num_distractors, num_workers=num_workers, device=device
    )
    #metrics
    writer = SummaryWriter(path)
    training_loss = torchmetrics.MeanMetric()
    training_acc = torchmetrics.Accuracy()
    test_loss = torchmetrics.MeanMetric()
    test_acc = torchmetrics.Accuracy()

    vocab_size = len(vocab['idx2word'])

    for episode in (p_bar := tqdm(range(num_episodes))):
        receiver.train()
        for all_features_batch, target_features_batch, target_captions_stack, target_idx_batch, ids_batch in data_loader_train:

            optimizer.zero_grad()

            all_features_batch = all_features_batch.to(device=device)
            target_captions_stack = target_captions_stack.to(device=device)
            target_idx_batch = target_idx_batch.to(device=device)
            target_idx_batch = torch.squeeze(target_idx_batch)

            logits = receiver(all_features_batch, target_captions_stack)
            loss = criterion(logits, target_idx_batch)
            loss.backward()
            optimizer.step()

            training_acc.update(logits.to('cpu'), target_idx_batch.to('cpu'))
            training_loss.update(loss.to('cpu'))

        receiver.eval()

        for all_features_batch, target_features_batch, target_captions_stack, target_idx_batch, ids_batch in data_loader_test:

            all_features_batch = all_features_batch.to(device=device)
            target_captions_stack = target_captions_stack.to(device=device)
            target_idx_batch = target_idx_batch.to(device=device)
            target_idx_batch = torch.squeeze(target_idx_batch)

            logits = receiver(all_features_batch, target_captions_stack)
            loss = criterion(logits, target_idx_batch)

            test_acc.update(logits.to('cpu'), target_idx_batch.to('cpu'))
            test_loss.update(loss.to('cpu'))

        scheduler.step()
        episode_training_loss = training_loss.compute()
        training_loss.reset()
        episode_training_acc = training_acc.compute()
        training_acc.reset()
        episode_test_loss = test_loss.compute()
        test_loss.reset()
        episode_test_acc = test_acc.compute()
        test_acc.reset()

        p_bar.set_description(
            f'Train: L{episode_training_loss :.3e} / ACC{episode_training_acc :.3e} || Test: L{episode_test_loss :.3e} / ACC{episode_test_acc :.3e}')
        writer.add_scalars(main_tag='todotag',
                           tag_scalar_dict={'training_loss': episode_training_loss, 'training_pp': episode_training_acc,
                                            'test_loss': episode_test_loss, 'test_pp': episode_test_acc},
                           global_step=episode)

    writer.flush()
    return receiver

def pretrain_receiver_lstm(receiver, num_episodes, path,  batch_size=128, num_distractors=7, lr=0.0001, num_workers=4, device='cpu'):
    receiver.to(device)
    receiver.train()
    optimizer = torch.optim.Adam(receiver.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    criterion = torch.nn.CrossEntropyLoss()

    (train_ds, test_ds), vocab = data.load_prepared_coco_data()

    data_loader_train = data.create_data_loader(
        train_ds, batch_size=batch_size, num_distractors=num_distractors, num_workers=num_workers, device=device
    )
    #split test data in val and test data
    test_val_cutoff = int(len(train_ds)*TEST_VAL_SPLIT)
    validation_ds = test_ds[test_val_cutoff:]
    test_ds = test_ds[:test_val_cutoff]
    data_loader_test = data.create_data_loader(
        test_ds, batch_size=batch_size, num_distractors=num_distractors, num_workers=num_workers, device=device
    )
    #metrics
    writer = SummaryWriter(path)
    training_loss = torchmetrics.MeanMetric()
    training_acc = torchmetrics.Accuracy()
    test_loss = torchmetrics.MeanMetric()
    test_acc = torchmetrics.Accuracy()

    vocab_size = len(vocab['idx2word'])

    for episode in (p_bar := tqdm(range(num_episodes))):
        receiver.train()
        for all_features_batch, target_features_batch, target_captions_stack, target_idx_batch, ids_batch in data_loader_train:

            optimizer.zero_grad()

            all_features_batch = all_features_batch.to(device=device)
            target_captions_stack = target_captions_stack.to(device=device)
            target_idx_batch = target_idx_batch.to(device=device)
            target_idx_batch = torch.squeeze(target_idx_batch)

            logits = receiver(all_features_batch, target_captions_stack)
            loss = criterion(logits, target_idx_batch)
            loss.backward()
            optimizer.step()

            training_acc.update(logits.to('cpu'), target_idx_batch.to('cpu'))
            training_loss.update(loss.to('cpu'))

        receiver.eval()

        for all_features_batch, target_features_batch, target_captions_stack, target_idx_batch, ids_batch in data_loader_test:

            all_features_batch = all_features_batch.to(device=device)
            target_captions_stack = target_captions_stack.to(device=device)
            target_idx_batch = target_idx_batch.to(device=device)
            target_idx_batch = torch.squeeze(target_idx_batch)

            all_features_batch = all_features_batch.to(device=device)
            target_captions_stack = target_captions_stack.to(device=device)
            target_idx_batch = target_idx_batch.to(device=device)

            logits = receiver(all_features_batch, target_captions_stack)
            loss = criterion(logits, target_idx_batch)

            test_acc.update(logits.to('cpu'), target_idx_batch.to('cpu'))
            test_loss.update(loss.to('cpu'))

        scheduler.step()
        episode_training_loss = training_loss.compute()
        training_loss.reset()
        episode_training_acc = training_acc.compute()
        training_acc.reset()
        episode_test_loss = test_loss.compute()
        test_loss.reset()
        episode_test_acc = test_acc.compute()
        test_acc.reset()

        p_bar.set_description(
            f'Train: L{episode_training_loss :.3e} / ACC{episode_training_acc :.3e} || Test: L{episode_test_loss :.3e} / ACC{episode_test_acc :.3e}')
        writer.add_scalars(main_tag='todotag',
                           tag_scalar_dict={'training_loss': episode_training_loss, 'training_pp': episode_training_acc,
                                            'test_loss': episode_test_loss, 'test_pp': episode_test_acc},
                           global_step=episode)

    writer.flush()
    return receiver

def interactive_training(sender, receiver, receiver_lr, sender_lr, path,  batch_size=128, num_distractors=7, num_episodes=200, num_workers=4, device='cpu', baseline_polyak=0.99):
    receiver.to(device)
    receiver.train()
    sender.to(device)
    sender.train()
    optimizer_receiver = torch.optim.Adam(receiver.parameters(), lr=receiver_lr)
    optimizer_sender = torch.optim.Adam(sender.parameters(), lr=sender_lr)

    scheduler_receiver = torch.optim.lr_scheduler.StepLR(optimizer_receiver, 1.0, gamma=0.95)
    scheduler_sender = torch.optim.lr_scheduler.StepLR(optimizer_sender, 1.0, gamma=0.95)

    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    baseline = torch.zeros(size=(1,)).to(device=device)

    (train_ds, test_ds), vocab = data.load_prepared_coco_data()

    data_loader_train = data.create_data_loader(
        train_ds, batch_size=batch_size, num_distractors=num_distractors, num_workers=num_workers, device=device
    )

    #split test data in val and test data
    test_val_cutoff = int(len(train_ds)*TEST_VAL_SPLIT)
    validation_ds = test_ds[test_val_cutoff:]
    test_ds = test_ds[:test_val_cutoff]
    data_loader_test = data.create_data_loader(
        test_ds, batch_size=batch_size, num_distractors=num_distractors, num_workers=num_workers, device=device
    )
    #metrics
    writer = SummaryWriter(path)
    training_loss = torchmetrics.MeanMetric()
    training_acc = torchmetrics.Accuracy()
    test_loss = torchmetrics.MeanMetric()
    test_acc = torchmetrics.Accuracy()
    entropy = torchmetrics.MeanMetric()
    probs = torchmetrics.MeanMetric()

    for episode in (p_bar := tqdm(range(num_episodes))):
        receiver.train()
        sender.train()
        for all_features_batch, target_features_batch, target_captions_stack, target_idx_batch, ids_batch in data_loader_train:

            optimizer_receiver.zero_grad()
            optimizer_sender.zero_grad()

            all_features_batch = all_features_batch.to(device=device)
            #target_captions_stack = target_captions_stack.to(device=device)
            target_idx_batch = target_idx_batch.to(device=device)
            target_encoded_features = target_distractor_encode_data(all_features_batch, target_idx_batch, num_distractors+1)#before squeezing!
            target_idx_batch = torch.squeeze(target_idx_batch)


            seq, log_p = sender(target_encoded_features, seq_data=None, device=device)
            seq = seq.detach() # technically not necessary but kind of nicer

            logits = receiver(all_features_batch, seq)
            loss = criterion(logits, target_idx_batch)

            #Receiver update (classic supervised learning)
            receiver_loss = torch.mean(loss)
            receiver_loss.backward()
            optimizer_receiver.step()

            #Sender update REINFORCE!
            log_p_mask = prob_mask(seq)
            log_p = log_p*log_p_mask
            #print(f'logpshape{log_p.size()}, maskshape{log_p_mask.size()}')
            log_p = torch.sum(log_p, dim=1)
            value = -loss.detach()
            baselined_value = value - baseline
            sender_reinforce_objective = log_p*baselined_value.detach()

            sender_loss = torch.mean(-sender_reinforce_objective)
            sender_loss.backward()
            optimizer_sender.step()

            avg_value = torch.mean(value)
            baseline = baseline_polyak*baseline + (1.-baseline_polyak)*avg_value

            training_loss.update(-torch.mean(value).to(device='cpu'))
            training_acc.update(logits.to(device='cpu'), target_idx_batch.to(device='cpu'))
            entropy.update(-torch.mean(log_p.to(device='cpu')))
            probs.update(torch.mean(torch.exp(log_p.to(device='cpu'))))

        #test
        sender.eval()
        #write one batch of test elems as sentences
        write_batch = True
        write_string = None
        sender.eval()
        receiver.eval()
        for all_features_batch, target_features_batch, target_captions_stack, target_idx_batch, ids_batch in data_loader_test:
            optimizer_receiver.zero_grad()
            optimizer_sender.zero_grad()

            all_features_batch = all_features_batch.to(device=device)
            target_captions_stack = target_captions_stack.to(device=device)
            target_idx_batch = target_idx_batch.to(device=device)
            target_encoded_features = target_distractor_encode_data(all_features_batch, target_idx_batch,
                                                                    num_distractors + 1)  # before squeezing!
            target_idx_batch = torch.squeeze(target_idx_batch)

            seq, log_p = sender(target_encoded_features, seq_data=None, device=device)
            seq = seq.detach()  # technically not necessary but kind of nicer
            if write_batch:
                argmax_seqs = seq.cpu().detach()
                test_sentences = utils.batch2sentences(vocab, argmax_seqs)
                org_sentences = utils.batch2sentences(vocab, target_captions_stack.cpu().detach())
                write_string = utils.summarize_test_sentences(org_sentences, test_sentences)
                write_batch = False

            logits = receiver(all_features_batch, seq)
            loss = criterion(logits, target_idx_batch)

            value = -loss.detach()
            test_loss.update(-torch.mean(value).to(device='cpu'))
            test_acc.update(logits.to(device='cpu'), target_idx_batch.to(device='cpu'))

        #end of episode stuff
        scheduler_receiver.step()
        scheduler_sender.step()
        episode_training_loss = training_loss.compute()
        training_loss.reset()
        episode_training_acc = training_acc.compute()
        training_acc.reset()
        episode_test_loss = test_loss.compute()
        test_loss.reset()
        episode_test_acc = test_acc.compute()
        test_acc.reset()
        episode_entropy = entropy.compute()
        entropy.reset()
        episode_probs = probs.compute()
        probs.reset()

        p_bar.set_description(
            f'Train: L{episode_training_loss :.3e} / ACC{episode_training_acc :.3e} || Test: L{episode_test_loss :.3e} / ACC{episode_test_acc :.3e} with H{episode_entropy:.3e} and p{episode_probs:.3e}')
        writer.add_scalars(main_tag='todotag',
                           tag_scalar_dict={'training_loss': episode_training_loss, 'training_pp': episode_training_acc,
                                            'test_loss': episode_test_loss, 'test_pp': episode_test_acc, 'entropy' : episode_entropy, 'probs':episode_probs},
                           global_step=episode)
        writer.add_text(tag='test_sentences', text_string=write_string, global_step=episode)


    writer.flush()








if __name__ == '__main__':
    num_distractors = 2

    device = torch.device('cuda:0')
    #agent = agents.transformer_sender_agent(2049, 2000, 32, 32, 2, 1,2,32,0.1)
    #pretrain_sender_transfomer(sender=agent, path=f'results/{datetime.now().strftime("%m_%d_%Y,%H:%M:%S")}', writer_tag='a_sender', batch_size=256, num_episodes=200, device=device)

    #agent = agents.lstm_sender_agent(feature_size=2049, text_embedding_size=16, vocab_size=2000, lstm_size=16, lstm_depth=1, feature_embedding_hidden_size=24)
    #print([param.size() for param in agent.parameters()])
    #pretrain_sender_lstm(sender=agent, path=f'results/{datetime.now().strftime("%m_%d_%Y,%H:%M:%S")}', writer_tag='a_sender', batch_size=256, num_episodes=200, device=device)

    #agent = agents.transformer_receiver_agent(2048, 2000, 128, 128, nhead=4, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=128, dropout=0.1)
    #pretrain_receiver_transformer(receiver=agent, num_episodes=100, path=f'results/{datetime.now().strftime("%m_%d_%Y,%H:%M:%S")}', writer_tag='a_t_receiver',  batch_size=128, num_distractors=7, lr=0.0001, device=device)

    #agent = agents.lstm_receiver_agent(feature_size=2048, text_embedding_size=64, vocab_size=2000, lstm_size=64, lstm_depth=2, feature_embedding_hidden_size=64, readout_hidden_size=32)
    #pretrain_receiver_lstm(receiver=agent, num_episodes=100, path=f'results/{datetime.now().strftime("%m_%d_%Y,%H:%M:%S")}', writer_tag='a_l_receiver',  batch_size=128, num_distractors=7, lr=0.0001, device=device)
    path = f'results/{datetime.now().strftime("%m_%d_%Y,%H:%M:%S")}/'
    sender = agents.lstm_sender_agent(feature_size=2049, text_embedding_size=128, vocab_size=2000, lstm_size=128, lstm_depth=2, feature_embedding_hidden_size=64)
    sender = pretrain_sender_lstm(sender=sender, path=path+'sender_pretraining', batch_size=256, num_distractors=num_distractors, num_episodes=20, device=device)
    receiver = agents.lstm_receiver_agent(feature_size=2048, text_embedding_size=128, vocab_size=2000, lstm_size=128, lstm_depth=2, feature_embedding_hidden_size=64, readout_hidden_size=32)
    receiver = pretrain_receiver_lstm(receiver=receiver, num_episodes=20,
                           path=path+'receiver_pretraining', batch_size=128, num_distractors=num_distractors, lr=0.0001, device=device)

    interactive_training(sender=sender, receiver=receiver, receiver_lr=0.00001, sender_lr=0.00001,path=path+'finetuning',  batch_size=1024, num_distractors=num_distractors,
                         num_episodes=200, device=device)



