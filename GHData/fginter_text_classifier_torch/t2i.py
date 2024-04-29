import json
import pickle
import itertools


def depth(data):
    #helper function, calculates the dimensionality depth
    if isinstance(data,int):
        return 0
    else:
        return 1+depth(data[0])

def maxima(data,level,maxima_list):
    maxima_list[level]=max(len(data),maxima_list[level])
    if isinstance(data[0],list):
        for d in data:
            maxima(d,level+1,maxima_list)
            
try:
    import torch

    def fill_torch(data,tensor,len_tensor,indices,tensor_size):
        #helper function
        # tensor -> data goes here, padded with zeros
        # len_tensor -> lengths go here
        if isinstance(data[0],int): #we hit the bottom, fill the data in
            assert len(tensor_size)==1 #last dimension, filling in the data
            data_v=torch.LongTensor(data[:tensor_size[0]])
            target=tensor[tuple(indices)]
            target[:data_v.size(0)]=data_v
            len_tensor[tuple(indices)]=data_v.size(0)
        else: #recursive call
            for i,d in itertools.islice(enumerate(data),tensor_size[0]):
                fill_torch(d,tensor,len_tensor,indices+[i],tensor_size[1:])

    def to_torch_long_tensor(data,max_seq_length=0):
        """
        Turns lists of python integers into torch tensor

        `data`: produced by t2i() and is thus a list (of lists, of lists...) of python integers
        `max_seq_length`: trim to this maximum. If 0, no trimming, if integer all dimensions (except the first one which is examples), will be trimmed to this length. If list of integers, every example's dimensions will be trimmed to those maxima.
        """
        maxima_list=[0]*depth(data) 
        maxima(data,0,maxima_list)
        #maxima_list is say [20000, 2386, 66] if our data is 20000 x 2386 x 66
        if max_seq_length!=0:
            if isinstance(max_seq_length,int):
                max_seq_length=[max_seq_length]*(len(maxima_list)-1)
            #now max_seq_length is something like [1000,66] or [55,55] meaning our tensor will be e.g. [20000,55,55]
            assert len(max_seq_length)==len(maxima_list)-1, "If max_seq_length is given as a list, it should have length of {} for this data, which has dimensionality {}.".format(len(maxima_list)-1,maxima_list)
            maxima_list=maxima_list[0:1]+list(min(actual,maximum) for actual,maximum in zip(maxima_list[1:],max_seq_length))
        out=torch.LongTensor(*maxima_list).zero_()
        out_len=torch.LongTensor(*maxima_list[:-1]).zero_() #the lengths are tensor of one less dimensionality
        fill_torch(data,out,out_len,[],list(out.size()))
        return out,out_len

    def torch_minibatched_2dim(data,batch_size):
        """
        data is tensor of (example X sequence_item)
        returns sequence_item X minibatch X example used by LSTM
        
        note - trimmed by batch_size
        """
        seq_count,seq_len=data.size()
        seq_count_mbatch_aligned=(seq_count//batch_size)*batch_size
        data_batched=data[:seq_count_mbatch_aligned].transpose(0,1).contiguous().view(seq_len,seq_count//batch_size,-1)
        return data_batched

    def torch_minibatched_1dim(data,batch_size):
        """
        data is tensor of (example)
        returns item X minibatch
        
        note - trimmed by batch_size
        """
        seq_count=data.size()[0]
        seq_count_mbatch_aligned=(seq_count//batch_size)*batch_size
        data_batched=data[:seq_count_mbatch_aligned].view(-1,seq_count//batch_size)
        return data_batched

        
    
except:
    pass #no torch, no need for to_torch_long_tensor
        


class T2I(object):

    def __init__(self,idict=None,with_padding="__PADDING__",with_unknown="__UNK__"):
        """
        `idict`: can be dictionary, string ending with ".json" or ".pkl/.pickle", or None
        `with_padding`: if not None, this string will be entry index 0
        `with_unknown`: if not None, this string will be entry index 1
        """

        self.padding=None
        self.unknown=None
        self.idict=None
        self.idict_rev=None #calculated only upon the first .reverse() call
        
        if isinstance(idict,str):
            if idict.endswith(".json"):
                with open(idict,"rt") as f:
                    self.idict,self.padding,self.unknown=json.load(f)
            elif idict.endswith(".pkl") or idict.endswith(".pickle"):
                with open(idict,"rb") as f:
                    self.idict,self.padding,self.unknown=pickle.load(f)
        elif isinstance(idict,dict):
            self.idict=idict
            self.padding=self.idict.get(with_padding)
            self.unknown=self.idict.get(with_unknown)
        elif idict is None:
            self.idict={}
            if with_padding is not None:
                self.padding=0
                self.idict[with_padding]=self.padding
            if with_unknown is not None:
                self.unknown=1
                self.idict[with_unknown]=self.unknown


    def save(self,name):
        """
        Save the dictionary
        `name`: string with file name, can end with .json or .pickle/.pkl
        """
        if name.endswith(".json"):
            with open(name,"wt") as f:
                json.dump((self.idict,self.padding,self.unknown),f)
        elif name.endswith(".pickle") or name.endswith(".pkl"):
            with open(name,"wb") as f:
                pickle.dump((self.idict,self.padding,self.unknown),f)
        else:
            raise ValueError("File type cannot be guessed from extension. Supported are .json .pkl .pickle.: "+self.idict)

    def __call__(self,inp,string_as_sequence=False,train=True):
        """
        Turn input to indices. Works also nested. If train is True, new entries are inserted into the dict, otherwise the unknown entry is used.
        
        `inp`: by default, a string is translated into single index, a sequence is translated into a list of outputs of a recursive call to self().
        `string_as_sequence`: if True, treat strings as sequences, effectively producing character level indices. self(["hi","there"]) will produce [2,3] if False, and [[2, 3], [4, 2, 5, 6, 5]] if True
        """
        if isinstance(inp,str):
            if train:
                if string_as_sequence:
                    return list(self.idict.setdefault(c,len(self.idict)) for c in inp)
                else:
                    return self.idict.setdefault(inp,len(self.idict))
            else: #not train
                if string_as_sequence:
                    return list(self.idict.get(c,self.unknown) for c in inp)
                else:
                    return self.idict.get(inp,self.unknown)
        else: #a list, must recurse
            return list(self(item,string_as_sequence,train) for item in inp)

    def reverse(self,inp):
        """
        Same as __call__ but in the opposite direction, i.e. index2string. Caches inversed dictionary on first call.
        """
        if self.idict_rev is None:
            self.idict_rev=dict((v,k) for k,v in self.idict.items())
        if isinstance(inp,int):
            return self.idict_rev[inp]
        else:
            return list(self.reverse(item) for item in inp)

        
if __name__=="__main__":
    t2i=T2I()
    print("string")
    print("hi",t2i("hi"))
    print("hi there hi",t2i("hi there hi".split()))
    print(t2i.idict)
    
    t2i_char=T2I()
    print()
    print("character")
    print("hi",t2i_char("hi",string_as_sequence=True))
    c=t2i_char("hi there hi".split(),string_as_sequence=True)
    print("hi there hi", c)
    print(t2i_char.idict)
    
          
    print()
    print("reverse")
    print(t2i_char.reverse(c))
    
