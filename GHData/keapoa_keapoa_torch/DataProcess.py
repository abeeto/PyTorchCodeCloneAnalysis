from torch.utils.data import Dataset
import torch
class Mydataset(Dataset):
    def __init__(self,data,tokenizer,maxlen):
        self.data = []
        self.tokenizer = tokenizer
        for i in range(len(data)):
            if len(data.iloc[i,0])+len(str(data.iloc[i,1])) <= maxlen - 3:
                input_ids = []
                segment_ids = []
                attention_mask = []
                input_ids.append(self.tokenizer.convert_tokens_to_ids("[CLS]"))
                segment_ids.append(0)

                for text in str(data.iloc[i,0]):
                    input_ids.append(self.tokenizer.convert_tokens_to_ids(text))
                    segment_ids.append(0)

                input_ids.append(self.tokenizer.convert_tokens_to_ids("[SEP]"))
                segment_ids.append(0)

                for text in str(data.iloc[i,1]):
                    input_ids.append(self.tokenizer.convert_tokens_to_ids(text))
                    segment_ids.append(1)

                input_ids.append(self.tokenizer.convert_tokens_to_ids("[SEP]"))
                segment_ids.append(1)
                attention_mask+=[1]*len(input_ids)

                assert len(input_ids)==len(segment_ids)
                assert len(input_ids)==len(attention_mask)
                # 填充成最大长度
                input_ids += [0] * (maxlen - len(input_ids))
                segment_ids += [0] * (maxlen - len(segment_ids))
                attention_mask += [0] * (maxlen - len(attention_mask))

                self.data.append([input_ids,segment_ids,attention_mask,data.iloc[i,2]])
    def __getitem__(self, index):
        return torch.tensor(self.data[index][0],dtype = torch.long),\
               torch.tensor(self.data[index][1],dtype = torch.long),\
               torch.tensor(self.data[index][2]),\
               self.data[index][3]
    def __len__(self):
        return len(self.data)
class standard_class_train_dataset(Dataset):
    def __init__(self,data,tokenizer,maxlen):
        self.data = []
        self.tokenizer = tokenizer
        for i in range(len(data)):

            inputs = self.tokenizer(text = str(data.iloc[i,0]),
                                    text_pair = str(data.iloc[i,1]),
                                    add_special_tokens = True,
                                    padding = False,#这个默认按照batch最大长度填充
                                    truncation = True,
                                    pad_to_max_length = True,#添加这个可自己设置最大长度
                                    max_length = maxlen,
                                    return_attention_mask = True)

            input_ids = inputs["input_ids"]
            segment_ids = inputs["token_type_ids"]
            attention_mask = inputs["attention_mask"]
            if i==0:
                print(len(input_ids),len(segment_ids),len(attention_mask))
                print(input_ids)
                print(segment_ids)
                print(attention_mask)
            assert len(input_ids)==maxlen,"the length of inputs_id is not equal to maxlen"
            assert len(attention_mask)==maxlen, "the length of attention_mask is not equal to maxlen"
            assert len(segment_ids)==maxlen, "the length of segment_ids is not equal to maxlen"


            self.data.append([input_ids,segment_ids,attention_mask,data.iloc[i,2]])
    def __getitem__(self, index):
        return torch.tensor(self.data[index][0],dtype = torch.long),\
               torch.tensor(self.data[index][1],dtype = torch.long),\
               torch.tensor(self.data[index][2]),\
               self.data[index][3]
    def __len__(self):
        return len(self.data)


class standard_class_test_dataset(Dataset):
    def __init__(self, data, tokenizer, maxlen):
        self.data = []
        self.tokenizer = tokenizer
        for i in range(len(data)):
            inputs = self.tokenizer(text=str(data.iloc[i, 0]),
                                    text_pair=str(data.iloc[i, 1]),
                                    add_special_tokens=True,
                                    padding=False,  # 这个默认按照batch最大长度填充
                                    truncation=True,
                                    pad_to_max_length=True,  # 添加这个可自己设置最大长度
                                    max_length=maxlen,
                                    return_attention_mask=True)

            input_ids = inputs["input_ids"]
            segment_ids = inputs["token_type_ids"]
            attention_mask = inputs["attention_mask"]
            assert len(input_ids) == len(segment_ids)
            assert len(input_ids) == len(attention_mask)

            self.data.append([input_ids, segment_ids, attention_mask])

    def __getitem__(self, index):
        return torch.tensor(self.data[index][0], dtype=torch.long), \
               torch.tensor(self.data[index][1], dtype=torch.long), \
               torch.tensor(self.data[index][2])

    def __len__(self):
        return len(self.data)
class testset(Dataset):
    def __init__(self,data,tokenizer,maxlen):
        self.data = []
        self.tokenizer = tokenizer
        for i in range(len(data)):

            input_ids = []
            segment_ids = []
            attention_mask = []
            input_ids.append(self.tokenizer.convert_tokens_to_ids("[CLS]"))
            segment_ids.append(0)

            for text in str(data.iloc[i,0]):
                input_ids.append(self.tokenizer.convert_tokens_to_ids(text))
                segment_ids.append(0)

            input_ids.append(self.tokenizer.convert_tokens_to_ids("[SEP]"))
            segment_ids.append(0)

            for text in str(data.iloc[i,1]):
                input_ids.append(self.tokenizer.convert_tokens_to_ids(text))
                segment_ids.append(1)

            input_ids.append(self.tokenizer.convert_tokens_to_ids("[SEP]"))
            segment_ids.append(1)
            attention_mask+=[1]*len(input_ids)

            assert len(input_ids)==len(segment_ids)
            assert len(input_ids)==len(attention_mask)
            # 填充成最大长度
            if len(input_ids) > maxlen:
                input_ids = input_ids[:(maxlen-1)] + [input_ids[-1]]
                segment_ids = segment_ids[:maxlen]
                attention_mask  = attention_mask[:maxlen]
            input_ids += [0] * (maxlen - len(input_ids))
            segment_ids += [0] * (maxlen - len(segment_ids))
            attention_mask += [0] * (maxlen - len(attention_mask))

            self.data.append([input_ids,segment_ids,attention_mask])
    def __getitem__(self, index):
        return torch.tensor(self.data[index][0],dtype = torch.long),\
               torch.tensor(self.data[index][1],dtype = torch.long),\
               torch.tensor(self.data[index][2])
    def __len__(self):
        return len(self.data)