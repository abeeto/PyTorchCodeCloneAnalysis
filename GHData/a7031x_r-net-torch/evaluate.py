import data
import func
import re
import string
import utils
from collections import Counter


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate_em_f1(feeder, tokens, predict_start, predict_end, target_starts, target_ends):
    predict = ' '.join(tokens[predict_start:predict_end+1])
    targets = [' '.join(tokens[start:end+1]) for start, end in zip(target_starts, target_ends)]
    em = metric_max_over_ground_truths(exact_match_score, predict, targets)
    f1 = metric_max_over_ground_truths(f1_score, predict, targets)
    return em, f1


def evaluate_batch(feeder, ids, y1p, y2p):
    total_em, total_f1, total = 0, 0, 0
    for id, predict_start, predict_end in zip(ids, y1p, y2p):
        eval_example = feeder.eval[id]
        tokens = eval_example['context_tokens']
        target_starts, target_ends = eval_example['y1s'], eval_example['y2s']
        em, f1 = evaluate_em_f1(feeder, tokens, predict_start, predict_end, target_starts, target_ends)
        total_em += em
        total_f1 += f1
        total += 1
    return total_em, total_f1, total


def evaluate_accuracy(model, dataset, batch_size=32, char_limit=16, size=None, output_file='./output/evaluate.txt', profile='dev'):
    model.eval()
    feeder = data.TrainFeeder(dataset, batch_size, char_limit)
    feeder.prepare(profile)
    size = size or feeder.size
    feeder.sort(size)
    lines = []
    total_em, total_f1, total = 0, 0, 0
    while feeder.cursor < size:
        ids, cs, qs, chs, qhs, y1s, y2s, ct, qt = feeder.next(batch_size)
        logits1, logits2 = model(func.tensor(cs), func.tensor(qs), func.tensor(chs), func.tensor(qhs), ct, qt)
        y1p, y2p = model.calc_span(logits1, logits2)
        for pids, qids, lable_start, label_end, predict_start, predict_end in zip(cs, qs, y1s, y2s, y1p, y2p):
            lines.append('--------------------------------')
            lines.append(feeder.ids_to_sent(pids))
            lines.append('question:  ' + feeder.ids_to_sent(qids))
            lines.append('reference: ' + feeder.ids_to_sent(pids[lable_start:label_end+1]))
            lines.append('predict:   ' + feeder.ids_to_sent(pids[predict_start:predict_end+1]))
        em, f1, bs = evaluate_batch(feeder, ids, y1p.tolist(), y2p.tolist())
        total_em += em
        total_f1 += f1
        total += bs
        print('{}/{}'.format(feeder.cursor, size))

    exact_match = total_em / total * 100
    f1 = total_f1 / total * 100
    message = 'EM: {:>.4F}, F1: {:>.4F}, Total: {}'.format(exact_match, f1, total)
    lines.append(message)
    utils.write_all_lines(output_file, lines)
    print('evauation finished with ' + message)
    return exact_match, f1


if __name__ == '__main__':
    import models
    import options
    import argparse

    def make_options():
        parser = argparse.ArgumentParser(description='train.py')
        options.model_opts(parser)
        options.evaluate_opts(parser)
        options.data_opts(parser)
        return parser.parse_args()

    opt = make_options()
    model, _ = models.load_or_create_models(opt, False)
    dataset = data.Dataset(opt)
    em, f1 = evaluate_accuracy(
        model,
        dataset,
        batch_size=opt.batch_size,
        char_limit=opt.char_limit,
        profile='test',
        output_file=opt.output_file)
    print('evaluation finished with EM: {}, F1: {}.'.format(em, f1))
