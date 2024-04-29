import os
from random import shuffle
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
import keras
import itertools
import torch.nn.functional as F
import torch.nn as nn
import scipy.sparse as sp

def read_data():
    all_data = []
    all_edges = []

    for root,dirs,files in os.walk('/home/datalab/data/cora'):
        for file in files:
            if '.content' in file:
                with open(os.path.join(root,file),'r') as f:
                    #print (f.read().splitlines())
                    all_data.extend(f.read().splitlines())
            elif 'cites' in file:
                with open(os.path.join(root,file),'r') as f:
                    all_edges.extend(f.read().splitlines())
                    
    #parse the data
    labels = []
    nodes = []
    X = []

    for i,data in enumerate(all_data):
        elements = data.split('\t')
        labels.append(elements[-1])
        X.append(elements[1:-1])
        nodes.append(elements[0])

    X = np.array(X,dtype=int)
    N = X.shape[0] #the number of nodes
    F = X.shape[1] #the size of node features
    # change the "0" "1" feature to float, otherwise pytorch cannot applied to int 
    new_X = []
    for i in range (N):
        temp = []
        for j in range(F):
            temp.append(float(X[i][j]))
        new_X.append(temp)
    X = new_X

    path="/home/datalab/data/cora/cora/cora"
    dataset=""

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    labels = encode_onehot(idx_features_labels[:, -1])
    labels = torch.LongTensor(np.where(labels)[1])
    X = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    X = normalize(X)
    X = torch.FloatTensor(np.array(X.todense()))


    #parse the edge
    edge_list=[]
    for edge in all_edges:
        e = edge.split('\t')
        edge_list.append((e[0],e[1]))
    num_classes = len(set(labels))
    return labels, nodes, X


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot
  
def limit_data(labels,limit=20,val_num=500,test_num=1000):
    """
    Get the index of train, validation, and test data
    
    label_counter = dict((l, 0) for l in labels)
    train_idx = []

    for i in range(len(labels)):
        label = labels[i]
        if label_counter[label]<limit:
            #add the example to the training data
            train_idx.append(i)
            label_counter[label]+=1
        
        #exit the loop once we found 20 examples for each class
        if all(count == limit for count in label_counter.values()):
            break
    
    #get the indices that do not go to traning data
    rest_idx = [x for x in range(len(labels)) if x not in train_idx]
    val_idx = rest_idx[:val_num]
    test_idx = rest_idx[val_num:(val_num+test_num)]
    """

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)
    return idx_train, idx_val,idx_test


def seperate_data(labels,limit=30,val_num=500,test_num=1000):
    '''
    Get the index of train, validation, and test data
    '''
    label_counter = dict((l, 0) for l in labels)
    train_idx = []

    for i in range(len(labels)):
        label = labels[i]
        if label_counter[label]<limit:
            #add the example to the training data
            train_idx.append(i)
            label_counter[label]+=1
        
        #exit the loop once we found 20 examples for each class
        if all(count == limit for count in label_counter.values()):
            break
    
    #get the indices that do not go to traning data
    rest_idx = [x for x in range(len(labels)) if x not in train_idx]
    val_idx = rest_idx[:val_num]
    test_idx = rest_idx[val_num:(val_num+test_num)]
    return train_idx, val_idx, test_idx


def get_train_test_data(train_idx, test_idx, input_data, label):
    input_train = []
    input_test = []
    output_train = []
    output_test = []
    for each in train_idx:
        input_train.append(input_data[each])
        output_train.append(label[each])

    for each in test_idx:
        input_test.append(input_data[each])
        output_test.append(label[each])
    return input_train, input_test, output_train, output_test


def encode_label(labels, class_label_list):
    hash_class= dict()
    temp = 0
    for i in range(len(class_label_list)):
        index = class_label_list[i]
        hash_class[index] = temp
        temp = temp + 1

    result = []
    for each_label in labels:
        result.append(hash_class[each_label])

    #label_encoder = LabelEncoder()
    #labels = label_encoder.fit_transform(labels)
    #print (labels[0:10])
    #enc = preprocessing.OneHotEncoder()
    # 2. FIT
    #enc.fit(labels)
    #labels = keras.utils.to_categorical(labels)
    return result

def accuracy(output, labels):
    #preds = output.max(1)[1].type_as(labels)
    #correct = preds.eq(labels).double()
    #correct = correct.sum()
    correct = 0
    predict = torch.max(output, 1).indices
    #print("emmmm?")
    #print(predict)
    #print(labels)
    for i in range(len(labels)):
        if (predict[i] == labels[i]):
            correct = correct + 1
    return correct / len(labels)

if __name__ == "__main__":
    torch.classes.load_library("build/libdcgan.so")
    print(torch.classes.loaded_libraries)

    graph_data_path = "/home/datalab/data/test3"
    # the number of node in the graph
    num_node = 2708
    input_feature_dim = 1433
    # the features have 5 dimensions
    net = torch.classes.my_classes.GCNWrap(input_feature_dim, 16, 7)
    manager1 = torch.classes.my_classes.ManagerWrap(0, num_node, graph_data_path)
    gview = torch.classes.my_classes.SnapWrap()
    manager1.create_static_view(gview)

    #adjacency_matrix = manager1.adj_matrix(gview)
    #print("wuhule")
    #print (adjacency_matrix)
    #print("adj_matrix")
    #torch.save(adjacency_matrix, 'file_cora_torch_0.pt') 


    #print ("1111")
    #print (net.parameters())

    # read the data
    labels, node_id, input_X = read_data()
    #print("2222")
    #print(len(input_X[0]))
    train_idx,val_idx,test_idx = limit_data(labels)
    input_train, input_test, output_train, output_test = get_train_test_data(train_idx, test_idx, input_X, labels)
    #print ("input_trAIN TENSOR")
    #print(len(input_X[0]))
    # seperate the data to train and test
    #print(set(input_train[0]))

    label_set = set(labels)
    class_label_list = []
    for each in label_set:
        class_label_list.append(each)
    #input_X = normalize(input_X)
    #input_X = torch.tensor(input_X)
    #output_train_label_encoded = encode_label(output_train, class_label_list)
    #print("train_label")
    #print(output_train_label_encoded)
    #output_test_label_encoded = encode_label(output_test, class_label_list)


    #print("00000")
    #print(output_train)

    labeled_nodes_train = torch.tensor(train_idx)  # only the instructor and the president nodes are labeled
    #labels_train = torch.tensor(output_train_label_encoded )  # their labels are different
    labeled_nodes_test = torch.tensor(test_idx)  # only the instructor and the president nodes are labeled
    #labels_test = torch.tensor(output_test_label_encoded)  # their labels are different

    #print("shazi")
    #print(labels_train.size())
    #print(len(output_test_label_encoded))
    #print(len(test_idx))


    # train the network
    optimizer = torch.optim.Adam(itertools.chain(net.parameters()), lr = 0.01, weight_decay = 5e-4)
    all_logits = []
    for epoch in range(200):
        # print("ai")
        #print(input_train.size())
        #logits = net.forward(input_X, gview)
        logits = net.forward(input_X, gview)

        # print("hu")
        # we save the logits for visualization later
        all_logits.append(logits.detach())
        logp = F.log_softmax(logits, 1)
        #print (logp)
        # we only compute loss for labeled nodes
        # print("ha??")
        #print(logp[labeled_nodes_train])

        loss = F.nll_loss(logp[labeled_nodes_train], labels[labeled_nodes_train])
        #make_dot(loss).render("torch_graph", format="png")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Epoch %d | Train_Loss: %.4f' % (epoch, loss.item()))

        # check the accuracy for test data
        logits_test = net.forward(input_X, gview)
        logp_test = F.log_softmax(logits_test, 1)

        acc_val = accuracy(logp_test[labeled_nodes_test], labels[labeled_nodes_test])
        print('Epoch %d | Test_accuracy: %.4f' % (epoch, acc_val))


# check the node predicton class
"""
print ("node prediction class")
for v in range(len(labeled_nodes_test)):
    temp = all_logits[199][v].numpy()
    cls = temp.argmax()
    print ("node" + str(v) + ":" + str(cls) + "\n")
"""
