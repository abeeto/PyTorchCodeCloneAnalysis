import torch
from torch.optim import Adam, lr_scheduler
import copy
import numpy as np
import logging
import pickle
import time
import os
import click

from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler
from train_event import pt_order,rewrite_content,extract
from utils import wrap,unwrap,wrap_X,unwrap_X
from GRU import Predict
from GRU import log_loss


n_features_embedding=7
n_hidden_embedding=40
n_features_rnn=40+4
n_hidden_rnn=10
logging.basicConfig(level=logging.INFO,format="[%(asctime)s %(levelname)s] %(message)s")

@click.command()
@click.argument("filename_train")
@click.argument("filename_test")
@click.argument("filename_model") ## this file will be generated after running train_event.py
@click.argument("n_events_train")
@click.argument("n_events_test")
@click.argument("filename_output")# this file lists predicted and test y values
@click.option("--n_jets_per_event", default=1)
def test(filename_train,filename_test,filename_model,n_events_train,n_events_test,filename_output,
         n_jets_per_event=1,random_state=1):
    
    logging.info("Calling with...")
    logging.info("\tfilename_train = %s" % filename_train)
    logging.info("\tfilename_test = %s" % filename_test)
    logging.info("\tfilename_model = %s" % filename_model)
    logging.info("\tn_events_train = %d" % n_events_train)
    logging.info("\tn_events_test = %d" % n_events_test)
    logging.info("\tfilename_output = %s" % filename_output)
    logging.info("\tn_jets_per_event = %d" % n_jets_per_event)
    
    logging.info("Loading train data")
    fd = open(filename_train, "rb")
    X = []
    y = []
    for i in range(n_events_train):
        e_i, y_i = pickle.load(fd,encoding='latin-1')
        
        four_features = []
        jets = []
        
        for j, (phi, eta, pt, mass, jet) in enumerate(e_i[:n_jets_per_event]):
            if len(jet["tree"]) > 1:
                four_features.append((phi, eta, pt, mass))
                jet = extract(permute_by_pt(rewrite_content(jet)), pflow=pflow)
                jets.append(jet)
    
        if len(jets) == n_jets_per_event:
            X.append([np.array(four_features), jets])
            y.append(y_i)

    y = np.array(y)
    
    fd.close()
    
    logging.info("\tfilename = %s" % filename_train)
    logging.info("\tX size = %d" % len(X))
    logging.info("\ty size = %d" % len(y))
    
    tf_features = RobustScaler().fit(np.vstack([features for features, _ in X]))
    tf_content = RobustScaler().fit(np.vstack([j["content"] for _, jets in X for j in jets]))

    ########################## Loading Test data######################
    X = None
    y = None
    logging.info("Loading test data")
    fd = open(filename_test, "rb")  # test file should be formatted like training file
    X = []
    y = []
    
    for i in range(n_events_test):
        e_i, y_i = pickle.load(fd,encoding='latin-1')
        four_features= []
        jets = []
        
        for j, (phi, eta, pt, mass, jet) in enumerate(e_i[:n_jets_per_event]):
            if len(jet["tree"]) > 1:
                four_features.append((phi, eta, pt, mass))
                jet = extract(permute_by_pt(rewrite_content(jet)), pflow=pflow)
                jets.append(jet)

        if len(jets) == n_jets_per_event:
           X.append([np.array(four_features), jets])
           y.append(y_i)
    
    y = np.array(y)
    
    fd.close()
    
    logging.info("\tfilename = %s" % filename_test)
    logging.info("\tX size = %d" % len(X))
    logging.info("\ty size = %d" % len(y))

    for i in range(len(X)):
        X[i][0] = tf_features.transform(X[i][0])
        
        for j in X[i][1]:
            j["content"] = tf_content.transform(j["content"])


############# Loading the model weights############################
        
    model = Predict(n_features_embedding, n_hidden_embedding, n_features_rnn, n_hidden_rnn,
                    n_jets_per_event)
    if torch.cuda.is_available():
        logging.warning("Moving model to GPU")
        model.cuda()
        logging.warning("Moved model to GPU")
    with open(filename_model, 'rb') as f:
        print("filename_model",filename_model) # this has to be in ".pt"
        state_dict = torch.load(f)
    model.load_state_dict(state_dict)
    model.eval()

############################Testing the model#################
    all_y_pred = []
    for start in range(0, len(y), 500):
        y_pred=model(X[start:start+500])
        all_y_pred.append(unwrap(y_pred))
    y_pred = np.concatenate(all_y_pred)

    # Save the predicted y values
    output = np.hstack((y.reshape(-1, 1),y_pred.reshape(-1, 1)))
    
    fd = open(filename_output, "wb")
    pickle.dump(output, fd, protocol=2)
    fd.close()


if __name__ == "__main__":
    test()
