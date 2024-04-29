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

from DNN import ThreeLayerNet
from DNN import log_loss



logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s %(levelname)s] %(message)s")

@click.command()
@click.argument("filename_train")
@click.argument("filename_model")
@click.argument("filename_test")
@click.argument("filename_output")
@click.option("--n_events", default=200000)
@click.option("--n_events_test", default=200000)
@click.option("--input_dim", default=4) ## should be always 4 if using 4 vectors or 4 features
@click.option("--hidden", default=64)
@click.option("--n_epochs", default=40)
@click.option("--batch_size", default=64)
@click.option("--step_size", default=0.005)
@click.option("--decay", default=0.8)
@click.option("--n_particles_per_event", default=6)
@click.option("--random_state", default=1)


os.environ['CUDA_VISIBLE_DEVICES'] = "1"
def train(filename_train,filename_model,filename_test,filename_output, n_events=200000, n_events_test=200000, input_dim=4,hidden=64, n_epochs=40,batch_size=64,step_size=0.05,decay=0.9,n_particles_per_event=6):
    
    logging.info("Calling with...")
    logging.info("\tfilename_train = %s" % filename_train)
    logging.info("\tfilename_model = %s" % filename_model) ## save the weights here
    logging.info("\tfilename_test = %s" % filename_test)
    logging.info("\tfilename_output = %s" % filename_output)
    logging.info("\tn_events = %d" % n_events)
    logging.info("\tn_events_test = %d" % n_events_test)
    logging.info("\tn_input_dim= %d" % input_dim)
    logging.info("\tn_hidden = %d" % hidden)
    logging.info("\tn_epochs = %d" % n_epochs)
    logging.info("\tbatch_size = %d" % batch_size)
    logging.info("\tstep_size = %f" % step_size)
    logging.info("\tdecay = %f" % decay)
    logging.info("\tn_particles_per_event = %d" % n_particles_per_event)
    
    ####################### Reading the train data #################################
    logging.info("Loading train data")
    
    fd = open(filename_train, "rb")# (e_i, y_i), where e_i is a list of (phi, eta, pt, mass) tuples.
    X = []
    y = []
    for i in range(n_events):
        v_i, y_i = pickle.load(fd,encoding='latin-1')
        v_i = v_i[:n_particles_per_event]
        X.append(v_i)
        y.append(y_i)

    y = np.array(y)
    fd.close()


    indices = torch.randperm(len(X)).numpy()[:n_events]
    X = [X[i] for i in indices]
    y = y[indices]

    # Preprocessing  # feature scaling
    logging.info("Preprocessing the train data")
    tf_features = RobustScaler().fit(np.vstack([features for features in X]))
    for i in range(len(X)):
        X[i] = tf_features.transform(X[i])
        
        if len(X[i]) < n_particles_per_event:
            X[i] = np.vstack([X[i],np.zeros((n_particles_per_event - len(X[i]), 4))]) ## padding

    # Split into train+validation
    logging.info("Splitting into train and validation...")
    X_train, X_valid, y_train, y_valid = train_test_split(X, y,test_size=5000, random_state=123)

###########################################Define MODEL ##############################

    logging.info("Initializing model...")
    model = ThreeLayerNet(input_dim, hidden, n_particles_per_event)
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
    
 ###############################VALIDATION OF DATA ########################################
    
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
    logging.info("Training...")
    iteration=1
    for i in range(n_epochs):
        logging.info("epoch = %d" % i)
        logging.info("step_size = %.4f" % step_size)
        t0 = time.time()
        for _ in range(n_batches):
            iteration += 1
            model.train()
            optimizer.zero_grad()
            start = torch.round(torch.rand(1) * (len(X_train) - batch_size)).numpy()[0].astype(np.int32)
            idx = slice(start, start+batch_size)
            X, y = X_train[idx], y_train[idx]
            y_var = wrap(y) ## wrap converts to torch tensor to run on GPU
            l = loss(model(X), y_var) ## X is a list. Inside "event_transform" class wrap_X is applied on X
            l.backward()
            optimizer.step()
            y = unwrap(y_var) ## unwrap y values
            callback(i, iteration, model) 
            t1 = time.time() 
        logging.info("Epoch took {} seconds".format(t1-t0))
        
        scheduler.step()
        step_size = step_size * decay


######################################### Testing #############################################

    logging.info("Testing...")
    fd = open(filename_test, "rb")
    X = []
    y = []
    for i in range(n_events_test):
        v_i, y_i = pickle.load(fd)
        v_i = v_i[:n_particles_per_event]
        X.append(v_i)
        y.append(y_i)

    y = np.array(y)
    fd.close()

    with open(filename_model, 'rb') as f:
        print("filename_model",filename_model) # this has to be in ".pt"
        state_dict = torch.load(f)
        model.load_state_dict(state_dict)
        model.eval()
    
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
    train()

