import os
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score
from scipy.stats import pearsonr
import numpy as np

def create_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
                
                
class Logger(object):
    def __init__(self, output_name):
        dirname = os.path.dirname(output_name)
        if not os.path.exists(dirname):
            os.mkdir(dirname)

        self.log_file = open(output_name, 'w')
        self.infos = {}

    def append(self, key, val):
        vals = self.infos.setdefault(key, [])
        vals.append(val)

    def log(self, extra_msg=''):
        msgs = [extra_msg]
        for key, vals in self.infos.iteritems():
            msgs.append('%s %.6f' % (key, np.mean(vals)))
        msg = '\n'.join(msgs)
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        self.infos = {}
        return msg

    def write(self, msg):
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        print(msg)
        
        
def get_roc_score(y_gt, y_pred, thr):
    from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score
    #y_gt [0.2, 0.3, 0.5, 0.7, ...]
    labels = np.array(y_gt[:])
    labels[labels>=thr] = 1
    labels[labels<thr] = 0
    score = roc_auc_score(labels, y_pred)
    return score


def get_pearson(y_gt, y_pred, with_p=True):
    coeff, p = pearsonr(y_gt, y_pred)
    if with_p:
        return coeff, p
    else:
        coeff


def get_inout_stats(in_probs, out_probs, labels_in=None):

    stats = {}
    if labels_in is not None:
        predicted_labels = np.argmax(in_probs, axis=1)
        labels_in = np.array(labels_in).squeeze()
        assert predicted_labels.shape == labels_in.shape
        stats['accuracy'] = np.sum(np.equal(predicted_labels, labels_in))\
                                  / float(len(labels_in))
    
    if len(np.shape(in_probs)) < 2:
        in_probs_max = in_probs
        out_probs_max = out_probs
    else:
        in_probs_max = np.max(in_probs, axis=1)
        out_probs_max = np.max(out_probs, axis=1)
    trues = np.append(np.ones(len(in_probs)), np.zeros(len(out_probs)))
    trues_flipped = np.append(np.zeros(len(in_probs)), np.ones(len(out_probs)))
    probs = np.append(in_probs_max, out_probs_max)
    fpr, tpr, thresholds = roc_curve(trues, probs)
    '''
    corrects_by_thresh = [len(in_probs_max[in_probs_max > thr])
                          + len(out_probs_max[out_probs_max < thr])
                          for thr in thresholds]
    '''
    stats['avg_in_max_softmax'] = np.mean(in_probs_max)
    stats['avg_out_max_softmax'] = np.mean(out_probs_max)
    stats['auroc'] = roc_auc_score(trues, probs)
    stats['aupr-in'] = average_precision_score(trues, probs)
    stats['aupr-out'] = average_precision_score(trues_flipped, probs)
    #stats['detection_accuracy'] = np.max(corrects_by_thresh) / float(len(probs))
    stats['fpr-at-tpr95'] = fpr[len(fpr) - len(tpr[tpr > 0.95])]

    return stats