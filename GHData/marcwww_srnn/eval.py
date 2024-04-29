import main
import torch
from params import args
import numpy as np
from params import EOS
import params
import data
BATCH_SIZE=args.batch_size
TEST_FILE=args.test_file

def train_accuracy(enc,dec):

    return accuracy(enc,dec,main.batch_pairs)

def accuracy(enc,dec,batch_pairs):

    num=0
    total=0

    def cmp(pre,tar):
        pre=list(pre)
        tar=list(tar)
        # print(pre[:len(tar)],tar)
        for i in range(len(tar)):
            if tar[i]!=pre[i]:
                return False

            if tar[i]==EOS:
                return True

    for i in range(len(batch_pairs)):
        src = batch_pairs[i][0]
        tar = batch_pairs[i][1]
        dec_tar = tar[1:, :]
        res, outputs = main.trans_one_batch(enc,dec,src)
        for i in range(BATCH_SIZE):
            if cmp(np.array((res[i])),dec_tar[:,i].cpu().numpy()):
               num+=1

        total+=BATCH_SIZE

    return num/total

def test_accuracy(enc,dec,fname):
    # pairs=[]
    # with open('data/'+name,'r') as f:
    #     for line in f:
    #         src,tar=line.strip('\n').split('\t')
    #         pairs.append((src,tar))
    input_lang, output_lang, pairs=data.prepareData(fname,reverse=False)

    batch_pairs=main.to_batch(main.input_lang,main.output_lang,pairs,BATCH_SIZE)
    return accuracy(enc,dec,batch_pairs)

if __name__ == '__main__':
    name='1525594849358548_stack'
    enc_file = args.output + '/' + 'enc_' + name + '.pt'
    dec_file = args.output + '/' + 'dec_' + name + '.pt'

    with open(enc_file, 'rb') as f:
        enc=torch.load(f,map_location=params.device_str)
    with open(dec_file, 'rb') as f:
        dec=torch.load(f,map_location=params.device_str)

    print(test_accuracy(enc,dec,TEST_FILE))




