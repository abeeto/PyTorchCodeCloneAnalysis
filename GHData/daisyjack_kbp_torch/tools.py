# coding: utf-8

import codecs
from itertools import izip

def out_comp_file():
    dev_f = codecs.open('data/cmn.txt', mode='rb', encoding='utf-8')
    result_f = codecs.open('bio_eva_result', mode='rb', encoding='utf-8')
    out = codecs.open('truth.txt', mode='wb', encoding='utf-8')
    for dev, result in izip(dev_f, result_f):
        dev_tokens = dev.strip().split('|||')[0].strip().split(' ')
        parts = result.strip().split('|||')
        truth = parts[0].strip().split(' ')
        rec = parts[1].strip().split(' ')
        truth_str = []
        rec_str = []
        for token, truth_label, rec_label in zip(dev_tokens, truth, rec):
            truth_str.append(token.split('#')[0] + '#{}'.format(truth_label))
            rec_str.append(token.split('#')[0] + '#{}'.format(rec_label))

        # out.write(' '.join(truth_str) + '~~|||~~~' + ' '.join(rec_str) + '\n')
        out.write(' '.join(truth_str) + '\n')

    out.close()
    dev_f.close()
    result_f.close()

def merge_tab_file():
    file0 = codecs.open('tab/all-mentions-0.tab', mode='rb', encoding='utf-8')
    file1 = codecs.open('tab/2016_eval_eng0.tab', mode='rb', encoding='utf-8')
    file_lst = [file0, file1]
    out_file = codecs.open('tab/merge.tab', mode='wb', encoding='utf-8')
    files_mention_set = []
    for i, f in enumerate(file_lst):
        if i == 0:
            mention_lst = []
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    mention = (parts[2], parts[3], parts[5], parts[6])
                    mention_lst.append(mention)
        else:
            mention_lst = []
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    mention = (parts[2], parts[3], parts[5], parts[6])
                    mention_lst.append(mention)
        files_mention_set.append(set(mention_lst))

    result = files_mention_set[0]
    for i in range(1, len(files_mention_set)):
        result = result.union(files_mention_set[i])

    for mention in result:
        out_file.write(u'ZJU\tEDL_0001019\t{}\t{}\tNIL\t{}\t{}\t1\n'.format(mention[0], mention[1], mention[2], mention[3]))


    file0.close()
    file1.close()
    out_file.close()


if __name__ == '__main__':
    merge_tab_file()
