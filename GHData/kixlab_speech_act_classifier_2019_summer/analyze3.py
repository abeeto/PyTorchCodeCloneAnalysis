import csv
import pdb
import pandas as pd
import argparse
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

parser = argparse.ArgumentParser(
    description='VRM Recognition Model Results Analysis')

parser.add_argument('--output_path',
                        help='test output for analysis')
parser.add_argument('--data_path', default='./data/data.csv',
                    help='dataset path')
parser.add_argument('--save_path',
                    help='path for saving analysis result')

args = parser.parse_args()


index_to_text = {
                -1: 'Ignore',
                0: 'Acknowledge (Backchannel)' ,
                1:'Other',
                2: 'Yes answers',
                3: 'Uninterpretable',
                4: 'Statement-opinion',
                5: 'Declarative Yes-No-Question',
                6: 'Agree/Accept',
                7: 'Rhetorical-Questions',
                8: 'Appreciation',
                9: 'No answers',
                10: 'Action-directive',
                11: '3rd-party-talk',
                12: 'Self-talk',
                13: 'Thanking',
                14: 'Summarize/reformulate',
                15: 'Yes-No-Question',
                16: 'Negative non-no answers',
                17: 'Other answers',
                18: 'Repeat-phrase',
                19: 'Hedge',
                20: 'Collaborative Completion',
                21: 'Signal-non-understanding',
                22: 'Declarative Wh-Question',
                23: 'Backchannel in question form',
                24: 'Open-Question',
                25: 'Conventional-opening',
                26: 'Statement-non-opinion',
                27: 'Affirmative non-yes answers',
                28: 'Non-verbal',
                29: 'Wh-Question',
                30: 'Conventional-closing',
                31: 'Downplayer',
                32: 'Offers, Options, Commits',
                33: 'Apology',
                34: 'Tag-Question',
                35: 'Dispreferred answers',
                36: 'Hold before answer/agreement',
                37: 'Maybe/Accept-part',
                38: 'Reject',
                39: 'Response Acknowledgement',
                40: 'Quotation',
                41: 'Or-Clause'
                }



if __name__=='__main__':

    of = pd.read_csv(args.output_path, error_bad_lines=False)
    val_targ = of["gt"]
    val_predict = of["pred"]
    val_conversation_id = of["conversation_id"]
    val_utterance_id = of["utterance_id"]

    assert(len(val_targ)==len(val_predict))

    _val_f1 = f1_score(val_targ, val_predict, average='micro')
    _val_recall = recall_score(val_targ, val_predict,average='micro')
    _val_precision = precision_score(val_targ, val_predict,average='micro')

    print("f1",_val_f1)
    print("recall",_val_recall)
    print("precision",_val_precision)

    #assert(len(ids)==len(val_targ))

    conversation_dict = {}

    for i in range(len(val_targ)):
        if val_conversation_id[i] not in conversation_dict.keys():
            conversation_dict[val_conversation_id[i]] = {}
        conversation_dict[val_conversation_id[i]][val_utterance_id[i]] = {'gt':val_targ[i],'pred':val_predict[i]}




    df = pd.read_csv(args.data_path, error_bad_lines=False)

    total_num = {}
    wrong_num = {}

    f = open(args.save_path, mode='w')
    wr = csv.writer(f)
    wr.writerow(['conversation_id','utterance_id','utterance','gt','pred','correct'])
    mode_dict = {}
    for i in range(len(df)):

        cid = df['conversation_id'][i]
        uid = df['utterance_id'][i]
        utterance = df['utterance'][i]
        #pdb.set_trace()
        if type(utterance) is not str:
            continue
        utterance = [word.split('/')[0] for word in utterance.strip().split(' ')]
        utterance = TreebankWordDetokenizer().detokenize(utterance)
        try:
            gt = conversation_dict[cid][uid]['gt']
            pred = conversation_dict[cid][uid]['pred']
        except:
            gt = '-1'
            pred = '-1'
        if gt not in mode_dict.keys():
            mode_dict[gt] = [0,0]

        if gt == pred:
            correct= 1
            mode_dict[gt][0] += 1
        else:
            correct = 0
        mode_dict[gt][1] += 1
        wr.writerow([cid,uid,utterance,index_to_text[int(pred)],index_to_text[int(gt)],correct])

    for key, rt in mode_dict.items():
        print( index_to_text[int(key)] + " : " + str(rt[0]) + "/"+ str(rt[1]) +" >> " + str(rt[0]/(rt[1]+0.000001)))



