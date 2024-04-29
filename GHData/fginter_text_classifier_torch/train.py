import argparse
import config
import data_imdb
import t2i
import model

import torch
import torch.autograd
from torch.autograd import Variable
import torch.nn as nn

import torch.optim as optim
import sys

import sklearn.feature_extraction
import time
import colorama
from colorama import Fore, Style



def minibatched_3dim(data,batch_size):
    seq_count,word_count,char_count=data.size()
    seq_count_mbatch_aligned=(seq_count//batch_size)*batch_size
    data_batched=data[:seq_count_mbatch_aligned].transpose(0,1).contiguous().view(word_count,seq_count//batch_size,-1,char_count)
    return data_batched

def prep_data(texts,classes,t2i_txt,t2i_chr,t2i_cls,is_test,start,batch_size):
    X_wrd=t2i_txt(texts,train=not is_test) #[[word01,word02,...],[word11,word12...]] (as integers)
    X_chr=t2i_chr(texts,train=not is_test,string_as_sequence=True) #[[[w,o,r,d,0,1],[w,o,r,d,0,2],..],[[w,o,r,d,1,1],[w,o,r,d,1,2],..]] (as integers)
    print("T2I completed at {:.1f}sec".format(time.clock()-start),file=sys.stderr)
    X_wrd_t,X_wrd_t_lengths=t2i.to_torch_long_tensor(X_wrd,[500]) #max 300 words per text
    X_chr_t,X_chr_t_lengths=t2i.to_torch_long_tensor(X_chr,[500,10]) #max 300 words per text, max 10 characters per word
    print("Torch tensors done at {:.1f}sec".format(time.clock()-start),file=sys.stderr)
    #print("X_wrd_t_lengths",X_wrd_t_lengths,file=sys.stderr)
    #print("X_chr_t_lengths",X_chr_t_lengths,file=sys.stderr)

    X_wrd_t_batched=t2i.torch_minibatched_2dim(X_wrd_t,batch_size)
    X_wrd_t_lengths_batched=t2i.torch_minibatched_1dim(X_wrd_t_lengths,batch_size)
    _,batches,_=X_wrd_t_batched.size()
    del X_wrd_t #no longer needed
    
    X_chr_t_batched=minibatched_3dim(X_chr_t,batch_size)
    X_chr_t_lengths_batched=t2i.torch_minibatched_2dim(X_chr_t_lengths,batch_size)
    del X_chr_t
    
    Y=t2i_cls(classes) #class indices
    Y_t_batched=torch.LongTensor(Y)[:batches*args.batch_size].view(batches,-1).contiguous() #minibatched
    return X_wrd_t_batched, X_wrd_t_lengths_batched, X_chr_t_batched, X_chr_t_lengths_batched, Y_t_batched
    

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--cpu', dest='cuda', default=True, action="store_false", help='Force CPU. Default is to use CUDA/GPU.')
    parser.add_argument('--batch-size',default=100, type=int, help='Batch size. Default %(default)d')
    args = parser.parse_args()

    tokenizer_func=sklearn.feature_extraction.text.CountVectorizer().build_tokenizer()
    
    config.set_cuda(args.cuda)
    
    torch.manual_seed(1)
    config.torch_mod.manual_seed(1) #cuda seed, if enabled

    start=time.clock()
    texts,classes=data_imdb.read_data("train",250) #X is list of texts
    texts_val,classes_val=data_imdb.read_data("test",2500) #X is list of texts
    texts=[tokenizer_func(t.lower()) for t in texts]
    texts_val=[tokenizer_func(t.lower()) for t in texts_val]

    t2i_txt=t2i.T2I() #word2index
    t2i_chr=t2i.T2I() #character2index
    t2i_cls=t2i.T2I(with_padding=None,with_unknown=None) #class2index

    print("Texts loaded at {:.1f}sec".format(time.clock()-start),file=sys.stderr)
    

    X_wrd_t_batched, X_wrd_t_batched_lengths, X_chr_t_batched, X_chr_t_batched_lengths,Y_t_batched=prep_data(texts,classes,t2i_txt,t2i_chr,t2i_cls,is_test=False,start=start,batch_size=args.batch_size)
    test_X_wrd_t_batched, test_X_wrd_t_batched_lengths, test_X_chr_t_batched, test_X_chr_t_batched_lengths, test_Y_t_batched=prep_data(texts_val,classes_val,t2i_txt,t2i_chr,t2i_cls,is_test=True,start=start,batch_size=args.batch_size)
    print("X_chr_t_batched size (wrdseq x batches x batchsize x charseq)",X_chr_t_batched.size(),file=sys.stderr)
    print("test_X_chr_t_batched size (wrdseq x batches x batchsize x charseq)",test_X_chr_t_batched.size(),file=sys.stderr)
    
    #Turn X_wrd_t to (seq X word)  and X_chr_t to (seq X word X char)

    
    network=model.TClass(len(t2i_cls.idict),wrd_emb_dims=(len(t2i_txt.idict),50),chr_emb_dims=(len(t2i_chr.idict),50))
    if config.cuda:
        network.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(network.parameters(), lr=0.1, momentum=0.6)
    
    batches=X_wrd_t_batched.size(1)
    print("X_wrd_t_batched size",X_wrd_t_batched.size())
    print("X_chr_t_batched size",X_chr_t_batched.size())
    print("Y_t_batched size",Y_t_batched.size())
    for iter in range(50):
        accum_loss=0
        accum_acc=0
        for batch_idx in range(batches):
            #print("Batch",batch_idx)
            minibatch_wrd_t=X_wrd_t_batched[:,batch_idx,:]
            minibatch_chr_t=X_chr_t_batched[:,batch_idx,:,:]

            minibatch_wrd_t_lengths=X_wrd_t_batched_lengths[:,batch_idx]
            minibatch_chr_t_lengths=X_chr_t_batched_lengths[:,batch_idx,:]
            #print("minibatch_t-size",minibatch_t.size())
            minibatch_wrd_tv=Variable(minibatch_wrd_t)
            minibatch_chr_tv=Variable(minibatch_chr_t)

            minibatch_wrd_tv_lengths=Variable(minibatch_wrd_t_lengths)
            minibatch_chr_tv_lengths=Variable(minibatch_chr_t_lengths)

            gold_classes_tv=Variable(Y_t_batched[batch_idx,:])
            if config.cuda:
                minibatch_wrd_tv=minibatch_wrd_tv.cuda()
                minibatch_chr_tv=minibatch_chr_tv.cuda()
                minibatch_wrd_tv_lengths=minibatch_wrd_tv_lengths.cuda()
                minibatch_chr_tv_lengths=minibatch_chr_tv_lengths.cuda()

                gold_classes_tv=gold_classes_tv.cuda()
                
            optimizer.zero_grad()
            outputs=network(minibatch_wrd_tv,minibatch_wrd_tv_lengths,minibatch_chr_tv,minibatch_chr_tv_lengths)
            #print("outputs",outputs)
            #print("gold",gold_classes_tv)
            values,indices=outputs.max(1)
            accum_acc+=float(torch.sum(indices.eq(gold_classes_tv)))/minibatch_wrd_t.size(1)
            loss=criterion(outputs,gold_classes_tv)
            #print("minibatch loss",float(loss))
            accum_loss+=float(loss)
            loss.backward()
            #print("STATEDICT",list(network.state_dict().keys()))
            #print("EMBGRAD",dict(network.named_parameters())["embedding.weight"].grad[:30])
            #print("DENSEGRAD",dict(network.named_parameters())["dense1.weight"].grad)
            #for p in network.parameters():
            #    print(p.grad)
            #print("LOSS",loss)
            #print()
            optimizer.step()
            #break
            #print("linout",network(minibatch_t))
        print("train loss",accum_loss/batches)
        print("train acc",accum_acc/batches*100)
        _,test_batches,_=test_X_wrd_t_batched.size()
        test_correct=0
        test_all=0
        for batch_idx in range(test_batches):
            test_minibatch_wrd_t=test_X_wrd_t_batched[:,batch_idx,:]
            test_minibatch_chr_t=test_X_chr_t_batched[:,batch_idx,:,:]
            test_minibatch_wrd_tv=Variable(test_minibatch_wrd_t)
            test_minibatch_chr_tv=Variable(test_minibatch_chr_t)

            test_minibatch_wrd_tv_lengths=Variable(test_X_wrd_t_batched_lengths[:,batch_idx])
            test_minibatch_chr_tv_lengths=Variable(test_X_chr_t_batched_lengths[:,batch_idx,:])

            test_gold_classes_tv=Variable(test_Y_t_batched[batch_idx,:])
            if config.cuda:
                test_minibatch_wrd_tv=test_minibatch_wrd_tv.cuda()
                test_minibatch_chr_tv=test_minibatch_chr_tv.cuda()
                test_minibatch_wrd_tv_lengths=test_minibatch_wrd_tv_lengths.cuda()
                test_minibatch_chr_tv_lengths=test_minibatch_chr_tv_lengths.cuda()
                test_gold_classes_tv=test_gold_classes_tv.cuda()
            test_outputs=network(test_minibatch_wrd_tv,test_minibatch_wrd_tv_lengths,test_minibatch_chr_tv,test_minibatch_chr_tv_lengths)
            test_values,test_indices=test_outputs.max(1)
            #print("batch_idx",batch_idx,torch.sum(test_indices.eq(test_gold_classes_tv)),file=sys.stderr)
            test_correct+=int(torch.sum(test_indices.eq(test_gold_classes_tv)))
            test_all+=test_minibatch_wrd_t.size(1)
        print(Fore.GREEN,"at {:.1f}sec".format(time.clock()-start),"Test acc",test_correct/test_all*100,"   ",test_correct,"/",test_all,Style.RESET_ALL,file=sys.stderr)
        print("\n"*2)
        
        

