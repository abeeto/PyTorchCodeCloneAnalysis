from torch.utils.data import Dataset,DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import random
import time
ap = argparse.ArgumentParser()
ap.add_argument("--translate", type=int,required=False, default=0,help="0:train, 1:translate, default=0")
ap.add_argument("--batch", type=int,required=False, default=100,help="batch size default=100")
ap.add_argument("--epoch", type=int,required=False, default=5,help="num of epoch default=5")
ap.add_argument("--lr", type=float,required=False, default=0.001,help="learning rate default=0.001")
ap.add_argument("--input", type=str,required=False, default='please input the sentence',help="input to translate default=test")
ap.add_argument("--gpu", type=int,required=False, default=1,help="1:gpu,other:cpu.")
args = vars(ap.parse_args())

if args['gpu']==1:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device=torch.device("cpu")
print(device)

TRANSLATE=int(args['translate'])
MAX_LEN=50      #限定最长句子
SOS_ID=1        #<sos>的ID
EOS_ID=2
BATCH_SIZE=int(args['batch'])    #batch大小
SEED=1234       #随机seed
EMB_DIM=1024      #嵌入维度
HID_DIM=1024
NUM_EPOCH=int(args['epoch'])
LEARNING_RATE=int(args['lr'])
INPUT=str(args['input'])
NUM_LAYERS=3    #隐层数量
INPUT_DIM=10000
OUTPUT_DIM=4000
DROPOUT=0.5
TEACHER_FORCING_RATIO=0.5
CLIP=1
SRC_VOCAB='./data/en.vocab'
TRG_VOCAB='./data/zh.vocab'
SRC_TRAIN_DATA='./data/en.number'
TRG_TRAIN_DATA='./data/zh.number'

class trainSet(Dataset):
    def __init__(self, train_file_en, train_file_zh):
        """实现初始化方法，在初始化的时候将数据读载入"""
        #self.df=pd.read_csv(csv_file)
        with open(train_file_en,'r') as f:
            lines=f.readlines()
            self.texts_en=[[[int(j) for j in i.strip().split()],len(i.strip().split())] for i in lines]
        with open(train_file_zh,'r') as f:
            lines=f.readlines()
            self.texts_zh=[[[int(j) for j in i.strip().split()],len(i.strip().split())] for i in lines]
    def __len__(self):
        '''
        返回df的长度
        '''
        return len(self.texts_en)
    def __getitem__(self, idx):
        '''
        根据 idx 返回一行数据
        '''
        #return self.df.iloc[idx].SalePrice
        return self.texts_en[idx],self.texts_zh[idx]

def collate_fn(batch_data,pad=0):
    """传进一个batch_size大小的数据"""
    # texta,textb,label = list(zip(*batch_data)) 输入多个序列时用该函数“解压”
    #texta[x][0]为数字序列  [x][1]为长度
    texta,textb=list(zip(*batch_data))
    textb_input=[[SOS_ID]+x[0][:-1] for x in textb]     #[<sos>,y1,y2...,yn]
    len_a=[x[1] for x in texta]
    len_b=[x[1] for x in textb]
    texta=[x[0] for x in texta]                         #[]
    textb=[x[0] for x in textb]

    # 删除过长句子和空句子 len==1 or len>MAX_LEN
    for i in range(len(len_a)-1,-1,-1):
        if (len_a[i]==1)|(len_a[i]>MAX_LEN)|(len_b[i]==1)|(len_b[i]>MAX_LEN):
            texta.pop(i)
            textb.pop(i)
            textb_input.pop(i)
            len_a.pop(i)
            len_b.pop(i)
    #padding
    maxlen_a = max(len_a)
    maxlen_b = max(len_b)
    texta = [x + [pad] * (maxlen_a - len(x)) for x in texta] #x=[seq,len]
    textb = [x + [pad] * (maxlen_b - len(x)) for x in textb]
    textb_input = [x + [pad] * (maxlen_b - len(x)) for x in textb_input]

    texta=torch.LongTensor(texta).to(device)
    len_a=torch.IntTensor(len_a).to(device)
    textb=torch.LongTensor(textb).to(device)
    textb_input=torch.LongTensor(textb_input).to(device)
    len_b=torch.IntTensor(len_b).to(device)
    # 返回padding后的输入源句、对应长度、目标输出、解码器输入、对应长度
    return (texta,len_a,textb,textb_input,len_b)



# Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        '''
        :param input_dim: 输入源词库的大小
        :param emb_dim:  输入单词Embedding的维度
        :param hid_dim: 隐层的维度
        :param n_layers: 几个隐层
        :param dropout:  dropout参数 0.5
        '''
        super().__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers = n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
    def forward(self, src):
        # src = [src sent len, batch size] 这句话的长度和batch大小 ,batch_first=True,维度颠倒
        embedded = self.dropout(self.embedding(src))
        # embedded = [src sent len, batch size, emb dim][batch,seqlen,embdim]
        outputs, (hidden, cell) = self.rnn(embedded)
        # x = [scr len, emb_dim]
        # w_xh = [emb_dim, hid_dim, n_layers]
        # scr sen len, batch size, hid dim, n directions, n layers
        # outputs: [src sent len, batch size, hid dim * n directions]   *[batch,seqlen,hiddim]
        # hidden, cell: [n layers* n directions, batch size, hid dim]   *[n layer,batch, hiddim]
        # outputs are always from the top hidden layer
        return hidden, cell



class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, input, hidden, cell):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]   *[n layer,batch, hiddim]
        # cell = [n layers * n directions, batch size, hid dim]     *[n layer,batch, hiddim]

        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hid dim]
        # context = [n layers, batch size, hid dim]

        input = input.unsqueeze(1)#[batch] -> [batch,1]
        #input = input.permute(1,0)#[1,batch]->[batch,1]
        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch size, emb dim]   *[batch,1,embdim]

        output, (hidden_, cell_) = self.rnn(embedded, (hidden, cell))

        # output = [sent len, batch size, hid dim * n directions]   *[batch,1(seqlen),hiddim]
        # hidden = [n layers * n directions, batch size, hid dim]   *[layer,batch,hiddim]
        # cell = [n layers * n directions, batch size, hid dim]     *[layer,batch,hiddim]
        prediction = self.out(output.squeeze(1))            #*output=[batch,hiddim]
        # prediction = [batch size, output dim]

        return prediction, hidden_, cell_       #源代码这里是hidden,cell,不确定不加self是否有问题


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    def forward(self, src, trg, trg_input, trg_size, teacher_forcing_ratio=0.5):
        # src = [src sent len, batch size]  *[batch,len]
        # trg = [trg sent len, batch size]  *[batch,len]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        batch_size = trg.shape[0]
        max_len = trg.shape[1]

        trg_vocab_size = self.decoder.output_dim
        # tensor to store decoder outputs
        outputs = torch.zeros(batch_size,max_len,
                              trg_vocab_size).to(self.device)#*[batch,len,vocab_size]

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder.forward(src)# shape:*[n layer, batch, hiddim]
        # first input to the decoder is the <sos> tokens
        input = trg_input[:, 0]
        for t in range(0, max_len):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder.forward(input, hidden, cell)#input=[sos,sos]
            # place predictions in a tensor holding predictions for each token
            outputs[:,t,:] = output     #*[batch,vocab_size]
            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            # get the highest predicted token from our predictions
            top1 = output.argmax(1)     #*[batch]

            # if teacher forcing, use actual next token as next input
            # if not,  use predicted token
            # 在 模型训练速度 和 训练测试差别不要太大 作一个均衡
            input = trg[:,t] if teacher_force else top1
        return outputs  #trg
    def forward_translate(self, src):
        #src=[seqlen]
        src=torch.LongTensor(src).unsqueeze(0).to(device)#[seqlen] -> [1(batch), seqlen]

        hidden, cell = self.encoder.forward(src)# shape:*[n layer, batch, hiddim]
        # first input to the decoder is the <sos> tokens
        outputs = torch.LongTensor([SOS_ID]).to(device)
        input = outputs
        for t in range(0, MAX_LEN):
            output, hidden, cell = self.decoder.forward(input, hidden, cell)#input=[sos] output=[batch,outdim]
            top1 = output.argmax(1)     #*[1(batch)]
            input = top1
            outputs = torch.cat((outputs, top1), 0)
            if top1 == 2:
                break
        return outputs


# init weights
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


# calculate the number of trainable parameters in the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def sequence_mask(lengths, max_len=None):
    #create a boolean mask from sequence lengths
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len, device = lengths.device).type_as(lengths)
            .unsqueeze(0).expand(batch_size,max_len)
            .lt(lengths.unsqueeze(1)))


def train(model, iterator, optimizer, criterion, clip, step):
    model.train()
    epoch_loss = 0

    for i, batch in enumerate(iterator,0):
        src, src_size, trg, trg_input, trg_size=batch
        #src = batch.src
        #trg = batch.trg
        optimizer.zero_grad()#梯度清零
        #forward(self, src, trg, trg_input, trg_size, teacher_forcing_ratio=0.5):
        output = model.forward(src, trg, trg_input, trg_size, TEACHER_FORCING_RATIO)

        # trg = [batch,trglen]
        # output = [batch, trglen, output dim]

        #mask matrix
        mask_matrix=sequence_mask(trg_size,trg.shape[1]).view(-1)
        #output = output[1:].view(-1, output.shape[-1])
        output = output.view(-1, output.shape[-1])  #[batch*trglen ,outdim]
        trg = trg.view(-1)                          #*[batch*trglen]


        loss = criterion(output, trg)
        loss=loss*mask_matrix
        cost=loss.sum()/mask_matrix.sum()

        cost.backward()   #反向传播
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)#梯度裁剪
        optimizer.step()#Adam优化

        epoch_loss += cost.item()
        step+=1

        if step%10==0:
            print("After %d steps, cost is %.3f"%(step,cost))

    return epoch_loss / len(iterator), step

def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output = model.forward(src, trg, 0)

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def translate(model, src_input):
    print('input sentence: ',src_input)
    src_input = src_input + ' <eos>'#加上结束标记
    with open(SRC_VOCAB,'r',encoding='utf-8') as vocab:
        src_vocab = [w.strip() for w in vocab.readlines()]
        src_id_dict = dict((src_vocab[x],x) for x in range(INPUT_DIM))
    test_en_ids = [(src_id_dict[en_text] if en_text in src_id_dict else src_id_dict['<unk>'])
                   for en_text in src_input.split()]
    print('input index :', test_en_ids)

    output = model.forward_translate(test_en_ids)
    print('output index: ', output)
    with open(TRG_VOCAB,'r',encoding='utf-8') as vocab:
        trg_vocab = [w.strip() for w in vocab.readlines()]
    output_text = ''.join([trg_vocab[x] for x in output[1:-1]])
    print('output text: ', output_text)


# calculate time
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def main():
    trainset = trainSet(train_file_en=SRC_TRAIN_DATA, train_file_zh=TRG_TRAIN_DATA)
    dataLoader = DataLoader(dataset=trainset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    encoder = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, NUM_LAYERS, DROPOUT)
    decoder = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, NUM_LAYERS, DROPOUT)
    seq2seq = Seq2Seq(encoder, decoder, device).to(device)
    seq2seq.apply(init_weights)
    # optimizer
    optimizer = optim.Adam(seq2seq.parameters())
    #344 657 483 381

    # index of <pad>
    #PAD_IDX = TRG.vocab.stoi['<pad>']
    # criterion
    # we ignore the loss whenever the target token is a padding token
    criterion = nn.CrossEntropyLoss(reduce=False)

    if TRANSLATE==0:
        best_valid_loss = float('inf')
        step=0
        for epoch in range(NUM_EPOCH):
            start_time = time.time()

            train_loss, step = train(seq2seq, dataLoader, optimizer, criterion, CLIP, step)
            #valid_loss = evaluate(model, valid_iterator, criterion)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            '''if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss'''
            torch.save(seq2seq.state_dict(), 'tut1-model.pt')
            print('saved')

            print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f}')
            #print(f'\t Val. Loss: {valid_loss:.3f}')
    else:
        print('translate:')
        seq2seq.load_state_dict(torch.load('tut1-model.pt'))

        seq2seq.eval()
        print('parameter loaded')
        translate(seq2seq,INPUT)



if __name__=='__main__':
    main()