def preprocessing(data):
    #get categorical data
    cat_data = data[[col for col in data.columns.tolist() if "Soil_Type" in col]]
    
    #reverse onehot on soil_type feature
    data['Soil'] = cat_data.dot(np.array(range(cat_data.columns.size))).astype(int)
    data.drop(columns = [col for col in data.columns.tolist() if "Soil_Type" in col])
    
    #make onehot label
    labels = np.zeros([len(file['Cover_Type']),7])
    for idx, d in enumerate(file['Cover_Type']):
        labels[idx][d-1] = 1
        
    data = data.drop('Cover_Type',axis = 1)
    
    #seperate categorical and numerical data
    cat_col = ['Soil']
    num_col = [col for col in data.columns if 'Soil' not in col]
    
    #scale numerical data
    scaler = StandardScaler()
    num_data = pd.DataFrame(scaler.fit(data[num_col]).transform(data[num_col]),columns = num_col)
    cat_data = data['Soil']
    data = pd.concat([num_data,cat_data],axis = 1)
    data[cat_col] = data[cat_col].astype('category')
    label = []
    for i in range(len(labels)):
        label.append(np.argmax(labels[i]))
    
    return data, num_col, cat_col, label

data, num_col, cat_col,labels = preprocessing(file)

embedding_dim = 4
vocab_size = len(set(np.unique(data[cat_col].values)))
context_size = len(data[cat_col])

class MyNet(nn.Module):
    def __init__(self, embedding_dim, vocab_size, context_size):
        super(MyNet, self).__init__()
        
        self.embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim = embedding_dim)

        self.linear1 = nn.Linear(18, 64)
        self.linear2 = nn.Linear(64,128)
        self.linear3 = nn.Linear(128,64)
        self.out = nn.Linear(128, 7)                     
        
    def forward(self,input):
        catout = self.embedding(torch.tensor(input[:,14]).long())
        catout = catout.view(-1,4)
        contout = torch.tensor(input[:,0:14]).float()
        contout = contout.view(-1,14)
        out = torch.cat((catout,contout), dim = 1)
        out = F.relu(self.linear1(out))
        out = F.relu(self.linear2(out))
        #out = F.relu(self.linear3(out))
        out = F.sigmoid(self.out(out))
        return out

def batch_generator(all_data , batch_size, shuffle=True):

    all_data = [np.array(d) for d in all_data]
    data_size = all_data[0].shape[0]
    print("data_size: ", data_size)
    if shuffle:
        p = np.random.permutation(data_size)
        all_data = [d[p] for d in all_data]

    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size > data_size:
            batch_count = 0
            if shuffle:
                p = np.random.permutation(data_size)
                all_data = [d[p] for d in all_data]
        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start: end] for d in all_data]

        
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

losses = []
loss = nn.CrossEntropyLoss()
#model = MyNet(embedding_dim,vocab_size,context_size)
optimizer = torch.optim.Adam(model.parameters())
batch_size, num_batch = 256, 2269
batch_gen= batch_generator([np.array(data),np.array(label)],batch_size,True)
start_time = time.time()
model.apply(init_weights)

for epoch in range(500):
    total_loss = 0
    k = 0
    time1 = time.time()
    for i in range(num_batch):
        
        batch_x, batch_y = next(batch_gen)
        batch_x, batch_y = np.array(batch_x,dtype = 'float64'),np.array(batch_y,dtype = 'float64')
        
        
        y_pred = model(torch.from_numpy(batch_x))
        output = loss(y_pred, torch.tensor(batch_y,dtype = torch.long))
        
        optimizer.zero_grad()
        output.backward()
        optimizer.step()
        total_loss += output.item()
        k+=1
        if k%1000==999:
            clear_output()
            print("--- epoch: %s ---" % epoch)
            print("--- batch: %s ---"  %k)
            #print("--- time elapsed: %s seconds ---" % (time.time() - start_time))
            #print("--- estimated time left for this epoch: %s seconds ---" % ((time.time() - time1)*((num_batch)**-1)*(581012-batch_size*k)))
            print("--- loss:%s --" %output.item())
            time1 = time.time()
    losses.append(total_loss/num_batch)
    print(epoch)
