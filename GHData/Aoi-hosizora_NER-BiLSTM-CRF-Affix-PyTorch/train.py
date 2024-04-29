import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
import torch
from torch import optim
from typing import Tuple, List, Dict

import dataset
from model import BiLSTM_CRF
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_train',     type=str, default='./data/eng.train')
    parser.add_argument('--dataset_val',       type=str, default='./data/eng.testa')
    parser.add_argument('--dataset_test',      type=str, default='./data/eng.testb')
    parser.add_argument('--pretrained_glove',  type=str, default='./data/glove.6B.100d.txt')
    parser.add_argument('--output_mapping',    type=str, default='./output/mapping.pkl')
    parser.add_argument('--output_affix_list', type=str, default='./output/affix_list.json')

    parser.add_argument('--use_crf',           type=int, default=1)
    parser.add_argument('--add_cap_feature',   type=int, default=1)
    parser.add_argument('--add_affix_feature', type=int, default=1)

    parser.add_argument('--use_gpu',     type=int, default=1)
    parser.add_argument('--model_path',  type=str, default='./model')
    parser.add_argument('--graph_path',  type=str, default='./output')
    parser.add_argument('--eval_path',   type=str, default='./evaluate/temp')
    parser.add_argument('--eval_script', type=str, default='./evaluate/conlleval.pl')

    args = parser.parse_args()
    args.use_crf = args.use_crf != 0
    args.add_cap_feature = args.add_cap_feature != 0
    args.add_affix_feature = args.add_affix_feature != 0
    args.use_gpu = args.use_gpu != 0
    args.use_gpu = args.use_gpu and torch.cuda.is_available()
    return args


def load_datasets(train_path: str, val_path: str, test_path: str, pretrained_glove: str, output_mapping: str, output_affix_list: str):
    train_sentences = dataset.load_sentences(train_path)[:14000]
    val_sentences = dataset.load_sentences(val_path)[:1500]  # <<<
    test_sentences = dataset.load_sentences(test_path)[-1500:]  # <<<

    dico_words, _, _ = dataset.word_mapping(train_sentences)
    _, char_to_id, _ = dataset.char_mapping(train_sentences)
    _, tag_to_id, id_to_tag = dataset.tag_mapping(train_sentences)
    _, word_to_id, _, word_embedding = dataset.load_pretrained_embedding(dico_words.copy(), pretrained_glove, word_dim=100)

    train_data = dataset.prepare_dataset(train_sentences, word_to_id, char_to_id, tag_to_id)
    val_data = dataset.prepare_dataset(val_sentences, word_to_id, char_to_id, tag_to_id)
    test_data = dataset.prepare_dataset(test_sentences, word_to_id, char_to_id, tag_to_id)
    prefix_dicts, suffix_dicts = dataset.add_affix_to_datasets(train_data, val_data, test_data)

    with open(output_mapping, 'wb') as f:
        mappings = {'word_to_id': word_to_id, 'tag_to_id': tag_to_id, 'char_to_id': char_to_id, 'word_embedding': word_embedding}
        pickle.dump(mappings, f)
    with open(output_affix_list, 'w') as f:
        json.dump([prefix_dicts, suffix_dicts], f, indent=2)

    print('Datasets status:')
    print('#train_data: {} / #val_data: {} / #test_data: {}'.format(len(train_data), len(val_data), len(test_data)))
    print('#word_to_id: {}, #char_to_id: {}, #tag_to_id: {}, #prefix_dicts: {}, #suffix_dicts: {}, '.format(len(word_to_id), len(char_to_id), len(tag_to_id), len(prefix_dicts), len(suffix_dicts)))
    print('#prefixes_2/3/4: [{}, {}, {}], #suffixes_2/3/4: [{}, {}, {}]'.format(len(prefix_dicts[1]), len(prefix_dicts[2]), len(prefix_dicts[3]), len(suffix_dicts[1]), len(suffix_dicts[2]), len(suffix_dicts[3])))
    return (train_data, val_data, test_data), (word_to_id, char_to_id, tag_to_id, id_to_tag), word_embedding, (prefix_dicts, suffix_dicts)


def train(model: BiLSTM_CRF, device: str, train_data: List[dataset.Data], val_data: List[dataset.Data], test_data: List[dataset.Data], model_path: str, graph_path: str, **kwargs):
    start_timestamp = time.time()

    lr = 0.015
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    total_loss_log = 0
    total_loss_plot = 0
    losses_plot, accuracies_plots, f1scores_plots = [], [[], []], [[], []]

    train_count = 0
    epochs = 10
    batches = len(train_data)
    log_every = 100
    save_every = int(batches / 2)
    plot_every = 100
    eval_every = 700

    print('\nStart training, totally {} epochs, {} batches...'.format(epochs, batches))
    for epoch in range(0, epochs):
        for batch, index in enumerate(np.random.permutation(batches)):
            model.train()
            train_count += 1
            data = train_data[index]

            words_in = torch.LongTensor(data.words).to(device)
            chars_mask = torch.LongTensor(data.chars_mask).to(device)
            chars_length = data.chars_length
            chars_d = data.chars_d
            caps = torch.LongTensor(data.caps).to(device)
            tags = torch.LongTensor(data.tags).to(device)
            words_prefixes = torch.LongTensor(data.words_prefix_ids).to(device)
            words_suffixes = torch.LongTensor(data.words_suffix_ids).to(device)

            feats = model(words_in=words_in, chars_mask=chars_mask, chars_length=chars_length, chars_d=chars_d, caps=caps, words_prefixes=words_prefixes, words_suffixes=words_suffixes)
            loss = model.calc_loss(feats, tags) / len(data.words)
            total_loss_log += loss.item()
            total_loss_plot += loss.item()

            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            if train_count % log_every == 0:
                avg_loss_log = total_loss_log / log_every
                total_loss_log = 0
                print('{} Epoch: {}/{}, batch: {}/{}, train loss: {:.4f}, time: {}'.format(
                    utils.time_string(), epoch + 1, epochs, batch + 1, batches, avg_loss_log, utils.time_since(start_timestamp, (epoch * batches + batch) / (epochs * batches))))

            if train_count % plot_every == 0:
                avg_loss_plot = total_loss_plot / plot_every
                total_loss_plot = 0
                losses_plot.append(avg_loss_plot)

            if (train_count % save_every == 0) or (batch == batches - 1 and epoch == epochs - 1):
                torch.save(model, '{}/savepoint_epoch{}_batch{}.pth'.format(model_path, epochs + 1, batches + 1))

            if train_count % eval_every == 0:
                print('\n{} Evaluating on validating dataset (epoch {}/{}, batch {}/{})...'.format(utils.time_string(), epoch + 1, epochs, batch + 1, batches))
                acc1, _, _, f1_score1 = evaluate(model=model, device=device, dataset=val_data, **kwargs)
                print('\n{} Evaluating on testing dataset (epoch {}/{}, batch {}/{})...'.format(utils.time_string(), epoch + 1, epochs, batch + 1, batches))
                acc2, _, _, f1_score2 = evaluate(model=model, device=device, dataset=test_data, **kwargs)
                accuracies_plots[0].append(acc1)
                accuracies_plots[1].append(acc2)
                f1scores_plots[0].append(f1_score1)
                f1scores_plots[1].append(f1_score2)
                print("\nContinue training...")
        # end batch

        # Referred from https://github.com/ZhixiuYe/NER-pytorch.
        new_lr = lr / (1 + 0.05 * train_count / len(train_data))
        utils.adjust_learning_rate(optimizer, lr=new_lr)
    # end epoch

    end_timestamp = time.time()
    start_time_str = utils.time_string(start_timestamp)
    end_time_str = utils.time_string(end_timestamp)
    print('Start time: {}, end time: {}, totally spent time: {:d}min'.format(start_time_str, end_time_str, int((end_timestamp - start_timestamp) / 60)))

    with open("{}/plots.log".format(graph_path), 'w') as f:
        f.write("time: {}\n\n".format(end_time_str))
        f.write("loss:\n[{}]\n\n".format(', '.join([str(i) for i in losses_plot])))
        f.write("acc1:\n[{}]\n\n".format(', '.join([str(i) for i in accuracies_plots[0]])))
        f.write("acc2:\n[{}]\n\n".format(', '.join([str(i) for i in accuracies_plots[1]])))
        f.write("f1:\n[{}]\n\n".format(', '.join([str(i) for i in f1scores_plots[0]])))
        f.write("f2:\n[{}]\n\n".format(', '.join([str(i) for i in f1scores_plots[1]])))

    epochs = list(range(1, len(losses_plot) + 1))
    plt.plot(epochs, losses_plot)
    plt.legend(['Training'])
    plt.xlabel('Index')
    plt.ylabel('Loss')
    plt.savefig('{}/loss.pdf'.format(graph_path))

    plt.clf()
    epochs = list(range(1, len(accuracies_plots[0]) + 1))
    plt.plot(epochs, accuracies_plots[0], 'b')
    plt.plot(epochs, accuracies_plots[1], 'r')
    plt.legend(['eng.testa', 'eng.testb'])
    plt.xlabel('Index')
    plt.ylabel('Accuracy')
    plt.savefig('{}/acc.pdf'.format(graph_path))

    plt.clf()
    epochs = list(range(1, len(f1scores_plots[0]) + 1))
    plt.plot(epochs, f1scores_plots[0], 'b')
    plt.plot(epochs, f1scores_plots[1], 'r')
    plt.legend(['eng.testa', 'eng.testb'])
    plt.xlabel('Index')
    plt.ylabel('F1-score')
    plt.savefig('{}/f1-score.pdf'.format(graph_path))

    print("graphs have been saved to {}".format(graph_path))


def evaluate(model: BiLSTM_CRF, device: str, dataset: List[dataset.Data], tag_to_id: Dict[str, int], id_to_tag: Dict[int, str], eval_path: str, eval_script: str) -> Tuple[float, float, float, float]:
    prediction = []
    confusion_matrix = torch.zeros((len(tag_to_id) - 2, len(tag_to_id) - 2))

    model.eval()
    for data in dataset:
        words_in = torch.LongTensor(data.words).to(device)
        chars_mask = torch.LongTensor(data.chars_mask).to(device)
        chars_length = data.chars_length
        chars_d = data.chars_d
        caps = torch.LongTensor(data.caps).to(device)
        words_prefixes = torch.LongTensor(data.words_prefix_ids).to(device)
        words_suffixes = torch.LongTensor(data.words_suffix_ids).to(device)

        feats = model(words_in=words_in, chars_mask=chars_mask, chars_length=chars_length, chars_d=chars_d, caps=caps, words_prefixes=words_prefixes, words_suffixes=words_suffixes)
        _, predicted_ids = model.decode_targets(feats)
        for (word, true_id, pred_id) in zip(data.str_words, data.tags, predicted_ids):
            line = ' '.join([word, id_to_tag[true_id], id_to_tag[pred_id]])
            prediction.append(line)
            confusion_matrix[true_id, pred_id] += 1
        prediction.append('')

    eval_lines, acc, pre, rec, f1 = utils.evaluate_by_perl_script(prediction=prediction, eval_path=eval_path, eval_script=eval_script)
    print('Accuracy: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1-score: {:.4f}'.format(acc, pre, rec, f1))
    print('Detailed result:')
    for i, line in enumerate(eval_lines):
        print(line)
    print('Confusion matrix:')
    print(("{: >2}{: >9}{: >15}%s{: >9}" % ("{: >9}" * confusion_matrix.size(0))).format(
        "ID", "NE", "Total",
        *([id_to_tag[i] for i in range(confusion_matrix.size(0))] + ["Percent"])
    ))
    for i in range(confusion_matrix.size(0)):
        print(("{: >2}{: >9}{: >15}%s{: >9}" % ("{: >9}" * confusion_matrix.size(0))).format(
            str(i), id_to_tag[i], str(confusion_matrix[i].sum()),
            *([confusion_matrix[i][j] for j in range(confusion_matrix.size(0))] +
              ["%.3f" % (confusion_matrix[i][i] * 100. / max(1, confusion_matrix[i].sum()))])
        ))
    return acc, pre, rec, f1


def main():
    args = parse_args()
    device = 'cuda' if args.use_gpu else 'cpu'

    (train_data, val_data, test_data), (word_to_id, char_to_id, tag_to_id, id_to_tag), word_embedding, (prefix_dicts, suffix_dicts) = load_datasets(
        train_path=args.dataset_train,
        val_path=args.dataset_val,
        test_path=args.dataset_test,
        pretrained_glove=args.pretrained_glove,
        output_mapping=args.output_mapping,
        output_affix_list=args.output_affix_list,
    )

    model = BiLSTM_CRF(
        vocab_size=len(word_to_id),
        tag_to_id=tag_to_id,
        pretrained_embedding=word_embedding,
        word_embedding_dim=100,
        char_count=len(char_to_id),
        char_embedding_dim=50,
        cap_feature_count=4,
        cap_embedding_dim=10,
        prefix_counts=[len(prefix_dicts[1]) + 1, len(prefix_dicts[2]) + 1, len(prefix_dicts[3]) + 1],
        suffix_counts=[len(suffix_dicts[1]) + 1, len(suffix_dicts[2]) + 1, len(suffix_dicts[3]) + 1],
        prefix_embedding_dims=[16, 16, 16],
        suffix_embedding_dims=[16, 16, 16],
        char_lstm_hidden_size=25,
        output_lstm_hidden_size=200,
        dropout_p=0.5,
        device=device,
        use_crf=args.use_crf,
        add_cap_feature=args.add_cap_feature,
        add_affix_feature=args.add_affix_feature,
    )
    model.to(device)
    train(
        model=model,
        device=device,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        model_path=args.model_path,
        graph_path=args.graph_path,
        **{
            'tag_to_id': tag_to_id,
            'id_to_tag': id_to_tag,
            'eval_path': args.eval_path,
            'eval_script': args.eval_script,
        },
    )


if __name__ == '__main__':
    main()
