import torch
from torch.optim import Adam, lr_scheduler
import copy
import numpy as np
import logging
import pickle
import time
import os
import click

from utils import wrap, unwrap, wrap_X, unwrap_X
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler
from GRU import Predict
from GRU import log_loss


def _pt(v):
    pz = v[2]
    p = (v[0:3] ** 2).sum() ** 0.5
    eta = 0.5 * (np.log(p + pz) - np.log(p - pz))
    pt = p / np.cosh(eta)
    return pt

def rewrite_content(jet):
    jet = copy.deepcopy(jet)
    content = jet["content"]
    tree = jet["tree"]
    
    def _rec(i):
        if tree[i, 0] == -1:
            pass
        else:
            _rec(tree[i, 0])
            _rec(tree[i, 1])
            c = content[tree[i, 0]] + content[tree[i, 1]]
            content[i] = c

_rec(jet["root_id"])

return jet

logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(levelname)s] %(message)s")

@click.command()
@click.argument("filename_train")
@click.argument("filename_model")
@click.option("--n_events", default=60000)
@click.option("--n_features_embedding", default=7)
@click.option("--n_hidden_embedding", default=40)
@click.option("--n_features_rnn", default=40+4)
@click.option("--n_hidden_rnn", default=15)
@click.option("--n_epochs", default=25)
@click.option("--batch_size", default=128)
@click.option("--step_size", default=0.005)
@click.option("--decay", default=0.8)
@click.option("--n_jets_per_event", default=4)


os.environ['CUDA_VISIBLE_DEVICES'] = "1" ## create the environment


def train(filename_train,filename_model,n_events=60000,n_features_embedding=7,
          n_hidden_embedding=40,n_features_rnn=40+4,n_hidden_rnn=10,
          n_epochs=25,batch_size=128,step_size=0.00, decay=0.7,n_jets_per_event=4):
    
    logging.info("Calling with...")
    logging.info("\tfilename_train = %s" % filename_train)
    logging.info("\tfilename_model = %s" % filename_model)
    logging.info("\tn_events = %d" % n_events)
    logging.info("\tn_features_embedding = %d" % n_features_embedding)
    logging.info("\tn_hidden_embedding = %d" % n_hidden_embedding)
    logging.info("\tn_features_rnn = %d" % n_features_rnn)
    logging.info("\tn_hidden_rnn = %d" % n_hidden_rnn)
    logging.info("\tn_epochs = %d" % n_epochs)
    logging.info("\tbatch_size = %d" % batch_size)
    logging.info("\tstep_size = %f" % step_size)
    logging.info("\tdecay = %f" % decay)
    logging.info("\tn_jets_per_event = %d" % n_jets_per_event)
    
    ####################### Reading the train data #################################
    logging.info("Loading train data")
    
    fd = open(filename_train, "rb")# (e_i, y_i) where e_i =(phi, eta, pt, mass, jet)
    X = []
    y = []
    for i in range(n_events):
        e_i, y_i = pickle.load(fd,encoding='latin-1')
        four_features = []
        jets = []
        
        for j, (phi, eta, pt, mass, jet) in enumerate(e_i[:n_jets_per_event]):
            if len(jet["tree"]) > 1:
                four_features.append((phi, eta, pt, mass))
                jet = extract(permute_by_pt(rewrite_content(jet)))
                jets.append(jet)
        if len(jets) == n_jets_per_event:
            X.append([np.array(four_features), jets])
            y.append(y_i)

    y = np.array(y)
    fd.close()
    
    indices = torch.randperm(len(X)).numpy()[:n_events]
    X = [X[i] for i in indices]
    y = y[indices]
    print("\tfilename = %s" % filename_train)
    print("\tX size = %d" % len(X))
    print("\ty size = %d" % len(y))


    # Preprocessing  # feature scaling
    logging.info("Preprocessing the train data")
    tf_features = RobustScaler().fit(np.vstack([features for features, _ in X]))
    tf_content = RobustScaler().fit(np.vstack([j["content"] for _, jets in X for j in jets]))
    for i in range(len(X)):
        X[i][0] = tf_features.transform(X[i][0])
        for j in X[i][1]:
            j["content"] = tf_content.transform(j["content"])

    # Split into train+validation
    logging.info("Splitting into train and validation")
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y,test_size=20000,random_state=42)

    ###########################################Define MODEL ##############################
    
    logging.info("Initializing model...")
    model = Predict(n_features_embedding, n_hidden_embedding, n_features_rnn, n_hidden_rnn, n_jets_per_event)
    if torch.cuda.is_available():
        logging.warning("Moving model to GPU")
        model.cuda()
        logging.warning("Moved model to GPU")

###########################OPTIMIZER AND LOSS ##########################################

    logging.info("Building optimizer...")
    optimizer = Adam(model.parameters(), lr=step_size)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=decay)

    n_batches = int(len(X_train) // batch_size)
    best_score = [-np.inf]
    best_model_state_dict = copy.deepcopy(model.state_dict())  # intial parameters of model

    def loss(y_pred, y):
        l = log_loss(y, y_pred.squeeze(1)).mean()
        return l
        
###############################VALIDATION OF DATA ########################################
    def callback(epoch, iteration, model):
        
        if iteration % n_batches == 0:
            model.eval()
            offset = 0; train_loss = []; valid_loss = []
            yy, yy_pred, accuracy_train, accuracy_valid = [], [],[],[]
            for i in range(len(X_valid) // batch_size):
                idx = slice(offset, offset+batch_size)
                Xt, yt = X_train[idx], y_train[idx]
                y_var = wrap(yt)
                tl = unwrap(loss(model(Xt), y_var)); train_loss.append(tl)
                
                
                Xv, yv = X_valid[idx], y_valid[idx]
                y_var = wrap(yv)
                y_pred = model(Xv)
                vl = unwrap(loss(y_pred, y_var)); valid_loss.append(vl)
                
                yv = unwrap(y_var); y_pred = unwrap(y_pred)
                yy.append(yv); yy_pred.append(y_pred)
                y_pred=np.column_stack(y_pred).ravel()
                accuracy_valid.append(np.sum(np.rint(y_pred)==yv)/float(len(yv)))
                offset+=batch_size
        
            train_loss = np.mean(np.array(train_loss))
            valid_loss = np.mean(np.array(valid_loss))
            accuracy_valid=np.mean(np.array(accuracy_valid))
            print("accuracy_valid:",accuracy_valid)
            print("train_loss:",train_loss)
            print("valid_loss:",valid_loss)
            roc_auc = roc_auc_score(np.column_stack(yy).ravel(), np.column_stack(yy_pred).ravel())
            print("roc_auc:",roc_auc)
            if roc_auc > best_score[0]:
               best_score[0]=roc_auc
               best_model_state_dict[0] = copy.deepcopy(model.state_dict())
               with open(filename_model, 'wb') as f:
                    torch.save(best_model_state_dict[0], f)
            
            scheduler.step(valid_loss)
            model.train()

###############################TRAINING ########################################
    logging.warning("Training the data")
    iteration=1
    for i in range(n_epochs):
        print("epoch = %d" % i)
        print("step_size = %.4f" % step_size)
        t0 = time.time()
        for _ in range(n_batches): ## mini batch
            iteration += 1
            model.train()
            optimizer.zero_grad()
            start = torch.round(torch.rand(1) * (len(X_train) - batch_size)).numpy()[0].astype(np.int32)
            idx = slice(start, start+batch_size)
            X, y = X_train[idx], y_train[idx]
            y_var = wrap(y) ## wrap converts to torch tensor to run on GPU
            l = loss(model(X), y_var) ## X is a list. Inside "GRU" class wrap_X is applied on X
            l.backward()
            optimizer.step()
            y = unwrap(y_var) ## unwrap y values
            callback(i, iteration, model) 
            t1 = time.time() 
        print(f'Epoch took {t1-t0} seconds')
        scheduler.step()
        step_size = step_size * decay


if __name__ == "__main__":
    train()
