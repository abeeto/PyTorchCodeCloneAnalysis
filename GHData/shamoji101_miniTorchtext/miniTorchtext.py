from copy import deepcopy
import random

class TextDataObject:
    
    def __init__(self, file:str, transform, min_word_num=None,Batch_size=1):
        self.Myinstance = self
        self.file_name = file
        self.vocab_dic={"<UNKNOWN>":0,"<PADDING>":1}
        self.vocab_dic_inv={}
        self.vocab_freq={"<UNKNOWN>":0,"<PADDING>":0}
        self.transform = transform
        self.processed_data=[]
        self.Batch_size=Batch_size
        
        with open(file,"r")as fp:
            for text in fp:
                
                processed_text,label = self._converter(text)
                self.processed_data.append((processed_text,label))
                
                #making vocab dic from processed_text
                
                for word in processed_text:
                    try:
                        self.vocab_freq[word]+=1
                    except KeyError:
                        self.vocab_freq[word]=1
                        self.vocab_dic[word]=len(self.vocab_dic)
        
        
        if min_word_num is not None:
            self.vocab_freq = {key:value for key,value in self.vocab_freq.items() if key == "<PADDING>" or key == "<UNKNOWN>" or value >= min_word_num}
            self.vocab_dic_inv = {vocab_id:key for vocab_id,(key,value) in enumerate(self.vocab_freq.items())}
            self.vocab_dic = {v:k for k,v in self.vocab_dic_inv.items()}
        else:
            self.vocab_dic_inv = {v:k for k,v in self.vocab_dic.items()}
        
        self.max_vocab = max(self.vocab_dic_inv.keys())
        
        
        
        
    def __getitem__(self,key):
        
        maxlen=0
        
        if type(key) is int:
            
            return TextDataObject.BatchObject(self, [self.processed_data[key]], len(self.processed_data[key][0]))
    
        else:
            
            for text,_ in self.processed_data[key]:
                maxlen=max(maxlen,len(text))

            return TextDataObject.BatchObject(self, self.processed_data[key], maxlen)
        
    def __len__(self):
        
        return len(self.processed_data)
    
    def convert(self, text:str,transform=None):
        
        if transform is None:
            transform = self.transform
        processed_text,label = self._converter(text,transform)
        maxlen=len(processed_text)
        return TextDataObject.BatchObject(self, [(processed_text,label)], maxlen)
    
    def _converter(self, text:str,transform=None):
        if transform is None:
            transform = self.transform
        try:
            processed_text,label=transform(text)
        except ValueError:
            raise ValueError("transform must return 2 data : processed text and label")
        try:
            assert type(processed_text) is list
        except AssertionError:
            raise ValueError("processed text return from transform must be word list.")
                
        return processed_text,label
    
    
    def deconvert(self, data):
        text=[]
        for word in data:
            text.append(self.vocab_dic_inv[word])
        return text
    
    def convert_from_file(self, file:str, transform=None, maxlen=0):
        
        if transform is None:
            transform = self.transform
        processed_data=[]
        with open(file,"r") as fp:
            for textline in fp:
                processed_data.append(self._converter(textline,transform))
        if maxlen == 0:
            for text,_ in processed_data:
                maxlen = max(maxlen,len(text))
                
        return TextDataObject.BatchObject(self, processed_data, maxlen)
        
    def GetFixedLengthBatch(self,maxlen):
        
        return TextDataObject.BatchObject(self, self.processed_data, maxlen)
    
    
    """
  if transform is None:
            transform = self.transform
        with open(file,"r")as fp:
            for textline in fp:
                processed_text,label=transform(textline)
    """          
        
            
        
    class BatchObject:
        def __init__(self,outer_instance, processed_data, maxlen):
            self.outer_instance = outer_instance
            self.processed_data = processed_data
            #print(self.processed_data)
            self.maxlen = maxlen #maxlen is fixed length
            self.text=[]
            self.label=[]
            self.fulldata = []
            self.processed_text_len_list = []
            self.batch_maxlen=0
            self._IterIndexPoint = 0
            self.batch_size = outer_instance.Batch_size
            for processed_text, label in processed_data:
                textdata=self.convert_with_dic(processed_text,self.outer_instance.vocab_dic)
                self.text.append(textdata)
                self.label.append(label)
                self.processed_text_len_list.append(len(processed_text))
                self.fulldata.append((textdata,label))
            
            if maxlen is None:
                self.batch_maxlen = max(self.processed_text_len_list)
            else:
                self.batch_maxlen = maxlen
            self.do_shuffle=True
                
        def __getitem__(self,key):
            
            #return self.outer_instance[key]
            
            if type(key) is int:
                return TextDataObject.BatchObject(self.outer_instance, [self.processed_data[key]], self.maxlen)
            else:
                return TextDataObject.BatchObject(self.outer_instance, self.processed_data[key], self.maxlen)
            
        
        def __add__(self,other):
            self.processed_data.extend(other.processed_data)
            return TextDataObject.BatchObject(self.outer_instance,self.processed_data, self.maxlen)
        
            
        def __len__(self):
        
            return len(self.processed_data)
        
        def __iter__(self):
            
            return deepcopy(self)
        
        
        def __next__(self):
            
            
            try:
                self[self._IterIndexPoint]
                next_batch_object=self[self._IterIndexPoint:self._IterIndexPoint+self.batch_size]
                if self.do_shuffle:
                    next_batch_object=next_batch_object.shuffle()
                self._IterIndexPoint+=self.batch_size
                return next_batch_object.text, next_batch_object.label

            except IndexError:
                
                raise StopIteration
                
        def convert_with_dic(self,processed_text,vocab_dic):
            
            data=[]
            for word,_ in zip(processed_text,range(self.maxlen)):
                try:
                    data.append(vocab_dic[word])
                except KeyError:
                    data.append(vocab_dic["<UNKNOWN>"])
                

            while len(data) < self.maxlen:
                data.append(vocab_dic["<PADDING>"])

            
            return data
        
        def split_validation(self,p=0.2):
            
            self.processed_data=sorted(self.processed_data, key=lambda k: random.random())
            split_point=int(len(self.processed_data)*p)
            return_A=self[:split_point]
            return_B=self[split_point:]
            return return_B,return_A #ここは、大きい方をTrain,小さい方をTestとするため。
        
        def BoW(self):
            allBoWvec=[]
            for wordlist in self.text:
                BoWvec=[0]*(self.outer_instance.max_vocab+1)
                for word in wordlist:
                    BoWvec[word]+=1
                allBoWvec.append(BoWvec)
            return allBoWvec, self.label
        
        def shuffle(self):
            
            self.processed_data = sorted(self.processed_data, key=lambda k: random.random())
            return self[:]
        

