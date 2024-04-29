import torch
import pickle
import pandas as pd
import numpy as np

from typing import List,Union,Optional,Dict

class SlotTokenizer:
    def __init__(self,data_list : List[str],embedding_table : Optional[pd.DataFrame] = None):
        '''
        Tokenizer for slot-based phoneme/grapheme representations.
        
        Args:
            data_list (List[str]) : word corpus from which to extract the token set.
            embedding_table (pd.Dataframe) : Maps each token to a prespecified vector embedding.
        '''
        
        ### Determine which values occur at each slot. We only assign tokens
        ### to these values.
        num_slots = len(data_list[0])
        slots = {slot:{} for slot in range(num_slots)}
        for word in data_list:
            for idx,char in enumerate(word):
                if char == '_':
                    continue;
                if slots[idx].get(char,False) is False:
                    slots[idx][char] = len(slots[idx])

        self.slots = {slot:slots[slot] for slot in slots if len(slots[slot])}

        self.embedding_table = embedding_table
        ### If [embedding_table] exists, we use its embedding vectors.
        if embedding_table is not None:
            self.embedding_size = len(slots) * len(embedding_table.columns)
        ### Otherwise, we create one-hot embeddings w/ dimensionality equal
        ### to the number of tokens.
        else:
            self.embedding_size = sum([len(slots[slot]) for slot in slots])

    def __call__(self,word:str) -> torch.Tensor:
        '''
        Tokenize [word] and map it to a vector.
        
        Args:
            word (str) : string to tokenize
        Returns:
            Vector embedding for [word]
        '''
        
        embedding = torch.zeros((self.embedding_size))
        marker = 0
        
        ### Iterative over characters in [word]
        for idx,char in enumerate(word):
            ### If an empty slot, continue.
            if idx not in self.slots:
                continue;

            ### If [embedding_table] exists, replace the current slot in [embedding]
            ### with the row of [embedding_table] corresponding to [char].
            if self.embedding_table is not None:
                embedding[marker:marker+len(self.embedding_table.columns)] = torch.FloatTensor(
                                                                        self.embedding_table.loc[char].to_numpy()
                                                                        )
                marker += len(self.embedding_table.columns)
            ### Otherwise, replace the current slot of [embedding] with a one-hot vector.
            else:
                if char != '_':
                    embedding[marker + self.slots[idx][char]] = 1 
                marker += len(self.slots[idx])
        return embedding


class Monosyllabic_Dataset(torch.utils.data.Dataset):
    def __init__(self,path_to_words,path_to_phon_mapping,path_to_sem,sample=True):
        '''
        Generic dataset for slot-based representations of
        monosyllabic words. 
        
        Args:
            path_to_words (str) : location of .csv file containing orthography and phonology.
            path_to_phon_mapping (str) : location of .csv file containing phonetic features.
            path_to_sem (str) : location of file (.npy or .npz) containing semantic embeddings. 
            
            sample (Optional[bool]) : If True, we sample words according to scaled
                                      frequency.
        '''
        super(Monosyllabic_Dataset,self).__init__()

        data = pd.read_csv(path_to_words).drop_duplicates()
        
        ### Parse orthography; create grapheme tokenizer
        self.orthography = data['ort']
        self.orthography_tokenizer = SlotTokenizer(self.orthography)

        ### Parse phonology; create phoneme tokenizer
        self.phonology = data['pho']
        phon_mapping = pd.read_csv(path_to_phon_mapping,sep="\t",header=None).set_index(0)
        self.phonology_tokenizer = SlotTokenizer(self.phonology,phon_mapping)
           
        ### Parse semantics
        semantics = torch.FloatTensor(np.load(path_to_sem)['data'])
        self.semantics = semantics[:,(semantics==0).any(dim=0)]

        ### Parse and scale word frequencies
        ### TODO: allow user to adjust frequency scaling
        self.frequencies = np.clip(np.sqrt(data['wf'])/(30000**.5),.05,1)
        self.frequencies = self.frequencies/np.sum(self.frequencies)
        
        self.sample = sample

    def __len__(self) -> int:
        return len(self.orthography)
    
    def __getitem__(self,idx : Union[int,str]) -> Dict[str,torch.Tensor]:
        
        if isinstance(idx,str):
           idx = self.orthography.index[self.orthography.apply(lambda x: x.replace('_','')) == idx][0]
            
        ### If [self.sample], sample from word corpus.
        if self.sample:
           idx = np.random.choice(np.arange(self.__len__()),p=self.frequencies)
        
        ### Get orthography, phonology, and semantics vectors
        orthography = self.orthography_tokenizer(self.orthography.iloc[idx])
        phonology = self.phonology_tokenizer(self.phonology.iloc[idx])
        semantics = self.semantics[idx]

        return {'orthography':orthography,'phonology':phonology,'semantics':semantics}
