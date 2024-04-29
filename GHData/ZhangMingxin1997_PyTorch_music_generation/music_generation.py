from music21 import * 
import numpy as np
import os
import torch
import time
import random
from torch import nn
from torch import optim
import torch.utils.data as Data
from torch.autograd import Variable
from imblearn.over_sampling import RandomOverSampler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class Rnn(nn.Module):
    def __init__(self):
        super(Rnn, self).__init__()
        self.lstm = nn.GRU(
            input_size=1,
            hidden_size=512,
            num_layers=3,
            batch_first=True,
            dropout=0.3
        )
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            # torch.nn.LeakyReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, n_vocab),
            torch.nn.Softmax()
        )

    def forward(self, x):
        out, (h_n) = self.lstm(x, None)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def read_midi(file):
    notes=[]
    notes_to_parse = None

    #解析MIDI文件
    midi = converter.parse(file)
    #基于乐器分组
    s2 = instrument.partitionByInstrument(midi)
    #遍历所有的乐器
    for part in s2.parts:
        #只选择钢琴
        #if 'Piano' in str(part): 
        notes_to_parse = part.recurse() 
        #查找特定元素是音符还是和弦
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                # print(element.offset)
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
    return notes

def convert_to_midi(prediction_output):
    offset = 0
    output_notes = []

    # 根据模型生成的值创建音符和和弦对象
    for pattern in prediction_output:
        # 模式是和弦
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # 模式是音符
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # 指定两个音符之间的持续时间
        offset += 0.5
        # offset += random.uniform(0.5,0.9)

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='output.mid')

music_path = "/home/zmx/music_generation/dataset/"
files=[i for i in os.listdir(music_path) if i.endswith(".mid")]
all_notes = []
for i in files:
    all_notes.append(read_midi(music_path + i))

notes = [element for notes in all_notes for element in notes]

#输入序列的长度
no_of_timesteps = 16

n_vocab = len(set(notes))  
pitch = sorted(set(item for item in notes))  

#为每个note分配唯一的值
note_to_int = dict((note, number) for number, note in enumerate(pitch))  

#准备输入和输出序列
X = []
y = []
for notes in all_notes:
    for i in range(0, len(notes) - no_of_timesteps, 1):
        input_ = notes[i:i + no_of_timesteps]
        output = notes[i + no_of_timesteps]
        X.append([note_to_int[note] for note in input_])
        y.append(note_to_int[output])

X = np.reshape(X, (len(X), no_of_timesteps, 1))
print(X.shape)
smo = RandomOverSampler(random_state=42)    # 处理样本数量不对称
nsamples, nx, ny = X.shape
d2_train_dataset = X.reshape((nsamples,nx*ny))
train_x, train_y = smo.fit_sample(d2_train_dataset, y)
train_x = train_x.reshape(len(train_x), nx, ny)
state = np.random.get_state()  # 打乱顺序
np.random.shuffle(train_x)
np.random.set_state(state)
np.random.shuffle(train_y)

#标准化输入
train_x = train_x / float(n_vocab) 

rnn = Rnn()
rnn.to(device)
optimizer = optim.Adam(rnn.parameters(), lr=0.0005)
scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma = 0.99)
criterion = nn.CrossEntropyLoss()
epochs = 500
batch_size= 64

data_x = torch.from_numpy(train_x)
data_x = torch.tensor(data_x, dtype=torch.float32)
data_y = torch.tensor(train_y)

torch_dataset = Data.TensorDataset(data_x, data_y) 
loader = Data.DataLoader( 
    dataset=torch_dataset, 
    batch_size=batch_size, # 批大小 
    shuffle=True, 
    num_workers=4, 
    ) 

model_path = "/home/zmx/music_generation/model/"
try:
    rnn.train(mode=True)
    loss_min = 100000
    for epoch in range(epochs):
        for i, (x, y) in enumerate(loader): 
            batch_x = Variable(x)
            batch_y = Variable(y)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            output = rnn(batch_x)
            loss = criterion(output,batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            TimeStr = time.asctime(time.localtime(time.time()))
        print('Epoch: {} --- {} --- '.format(epoch, TimeStr))
        print('learning rate: {}'.format(optimizer.param_groups[0]['lr']))
        print(loss)
        scheduler.step()

        if loss <= loss_min:
            torch.save(rnn, model_path + 'model.pkl')
            loss_min = loss

except KeyboardInterrupt:
     print('中断训练，直接生成')

finally:
    model = torch.load(model_path + 'model.pkl')
    model.eval()
    with torch.no_grad():
        start = np.random.randint(0, len(train_x)-1)
        pattern = train_x[start]
        int_to_note = dict((number, note) for number, note in enumerate(pitch))
        prediction_output = []
        for note_index in range(70):
            #输入数据
            input_ = np.reshape(pattern, (1, len(pattern), 1))
            input_ = torch.tensor(input_, dtype=torch.float32)
            input_ = input_.to(device)
            #预测并选出最大可能的元素
            proba = model(input_)
            input_ = input_.cpu()
            proba = proba.cpu()
            index = int(np.argmax(proba))
            pred = int_to_note[index]
            prediction_output.append(pred)
            pattern = np.append(pattern,index/float(n_vocab))
            #将第一个值保留在索引0处
            pattern = pattern[1:len(pattern)]

    convert_to_midi(prediction_output)