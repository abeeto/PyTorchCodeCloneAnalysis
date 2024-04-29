import config

import glob
import random
random.seed(1)

def read_files(file_list):
    texts=[]
    for f_name in file_list:
        with open(f_name) as f:
            txt=f.read().strip()
            texts.append(txt)
    return texts

def read_data(section,max_doc=0):
    """section is train or test"""
    pos_f=sorted(glob.glob("aclImdb/{}/pos/*.txt".format(section)))
    neg_f=sorted(glob.glob("aclImdb/{}/neg/*.txt".format(section)))
    if max_doc>0:
        pos_f=pos_f[:max_doc]
        neg_f=neg_f[:max_doc]
    data_pos=read_files(pos_f)
    data_neg=read_files(neg_f)
    classes=["pos"]*len(data_pos)+["neg"]*len(data_neg)
    data=data_pos+data_neg
    assert len(classes)==len(data)
    data_classes=list(zip(data,classes))
    random.shuffle(data_classes)
    return list(x[0] for x in data_classes),list(x[1] for x in data_classes)


    
    
