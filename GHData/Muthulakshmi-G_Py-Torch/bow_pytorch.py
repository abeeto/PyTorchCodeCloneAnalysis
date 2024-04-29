import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as grad


data=[("me gusta comer en la cafeteria".split(), "SPANISH"),("Give it to me".split(), "ENGLISH"),("No creo que sea una buena idea".split(), "SPANISH"),("No it is not a good idea to get lost at sea".split(), "ENGLISH")]


test_data=[("Yo creo que si".split(), "SPANISH"),("it is lost on me".split(), "ENGLISH")]




#ella words athu entha indexla erukuthunu pakkurom.
word_to_index={}#vocab la erukum ecah word-i unique #integerku map panrom
for sent,_ in data+test_data: 
    for word in sent: #sent-la erukum words
        if word not in word_to_index:#oru word  word indexla #illana
            word_to_index[word]=len(word_to_index)#antha particular word-#in length 
print(word_to_index)

#vocab,labels set pannurom
VOCAB_SIZE=len(word_to_index)
NUM_LABELS=2


class BoWClassifier(nn.Module):#here we use bag of words model
    def __init__(self,num_labels,vocab_size):
        super(BoWClassifier,self).__init__()
        self.linear=nn.Linear(vocab_size,num_labels)


    def forward(self,bow_vec):#passing the input throgh the linearlayer
        return F.log_softmax(self.linear(bow_vec),dim=1)#vectorsku softmax function

#bow vectors create pannrom
def make_bow_vector(sentence, word_to_ix):
    vec = torch.zeros(len(word_to_ix))#oru sentence la words length
    for word in sentence:
        vec[word_to_ix[word]] += 1
    return vec.view(1, -1)


def make_target(label, label_to_ix):
    return torch.LongTensor([label_to_ix[label]])

model=BoWClassifier(NUM_LABELS,VOCAB_SIZE)

#affine map a and b .intha paramters
for param in model.parameters():
    print(param)

with torch.no_grad():
    sample=data[0]
    bow_vector=make_bow_vector(sample[0],word_to_index)# to run model pass bow_vec
    log_probs=model(bow_vector)
    print(log_probs)

label_to_index={"SPANISH":0,"ENGLISH":1}

#TEST DATAVA RUN PANNU.TRAINKU MUNNADI
with torch.no_grad():
    for instance,label in test_data:
        bow_vec=make_bow_vector(instance,word_to_index)
        log_probs=model(bow_vec)
        print(log_probs)

print(next(model.parameters())[:,word_to_index["creo"]])

loss_function=nn.NLLLoss()

optimizer=optim.SGD(model.parameters(),lr=0.1)

for epoch in range(100):
    for instance,label in data:
        model.zero_grad()

        bow_vec=make_bow_vector(instance,word_to_index)
        target=make_target(label,label_to_index)

        log_probs=model(bow_vec)#run forward pass
#compute loss,gradients,update

        loss=loss_function(log_probs,target)
        loss.backward()
        optimizer.step()


with torch.no_grad():
    for instance,label in test_data:
        bow_vec=make_bow_vector(instance,word_to_index)
        log_probs=model(bow_vec)
        print(log_probs)


print(next(model.parameters())[:,word_to_index["creo"]])
