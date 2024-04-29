# Dev Note:
# Log Finished.
# early stop Finished.
# Autosave Finished.
# Learning rate decay
# Warm start
# Parameters regularization
# clean up the confusing torchvision.transforms functions.
# Provide interface in fit method to use user-defined series learning rate.
# Add learning rate callback.

import io
import os
import sys
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from sklearn.metrics import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

# from image_augmentation_tools_authors import *
from pytorch_flow_from_directories import *

class PyTorchBaseModel(object):

    def __init__(self, model=None, loss_function=None, optimizer=None, log_dir=None, checkpoint_dir=None):

        # Tic:
        self.tic = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        # Initialize log:
        self.__init_logging(log_dir=log_dir)

        # Define model checkpoints:
        self.checkpoint_dir = checkpoint_dir

        # Define model and some pre-checks:
        self.model = model
        if self.model is not None:
            assert hasattr(self.model, "forward"), "Input PyTorch model should be consistent to have 'forward' method."

        # Define loss function:
        self.loss = loss_function
        logging.info("Initial loss function: {}".format(loss_function))

        # Define optimizer:
        self.optimizer = optimizer
        logging.info("Initial optimizer: {}".format(optimizer))

    def __init_logging(self, log_dir=None):

        if log_dir is None:
            self._log_stream = io.StringIO()
            logging.basicConfig(
                stream=self._log_stream,
                level=logging.INFO,
                format="[%(asctime)s (Local Time)] %(levelname)s : %(message)s", # Local time may vary for cloud services.
                datefmt="%m/%d/%Y %I:%M:%S %p"
            )
        else:
            if not os.path.isdir(log_dir):
                os.makedirs(log_dir)

            _log_file_name = 'PyTorch_log_{}.txt'.format(self.tic)
            self.__log_file_dir = os.path.join(log_dir, _log_file_name)
            logging.basicConfig(
                filename=self.__log_file_dir,
                level=logging.INFO,
                format="[%(asctime)s (Local Time)] %(levelname)s : %(message)s",
                datefmt='%m/%d/%Y %I:%M:%S %p'
            )

        logging.info("Created log.")

    def get_log(self):
        if hasattr(self, "_log_stream"):
            return self._log_stream.get_value()
        else:
            try:
                with open(self.__log_file_dir, 'r') as log_file:
                    return log_file.read()
            except Exception as e:
                print('Failed in opening log file. Error: {}'.format(e))
                raise

    def pytorch_default_loss_functions(self):
        pass

    # Need to finish the loss function list:
    def set_loss_function(self, loss_function=None):
        if loss_function is None:
            logging.error("Error: No loss function was provided.")
            raise NotImplementedError("Need to define loss function.")
        else:
            if isinstance(loss_function, str):
                assert loss_function in (
                    "binary_cross_entropy", "nll_loss", "mse_loss", "cross_entropy", "binary_cross_entropy_with_logits"
                )
                logging.info("Use loss function: {}".format(loss_function))
                if loss_function.upper() == "NLL_LOSS":
                    return torch.nn.functional.nll_loss
                if loss_function.upper() == "BINARY_CROSS_ENTROPY":
                    return torch.nn.functional.binary_cross_entropy
                if loss_function.upper() == "BINARY_CROSS_ENTROPY_WITH_LOGITS":
                    return torch.nn.functional.binary_cross_entropy_with_logits
                if loss_function.upper() == "CROSS_ENTROPY":
                    return torch.nn.functional.cross_entropy
                    # print("Use loss function: {}".format(loss_function))

    # Need to finish the optimizer list:
    def set_optimizer(self, optimizer=None):
        # assert isinstance(learning_rate, float) and learning_rate>0, TypeError("Input paramter 'learning_rate' can only be positive float.")

        if optimizer is None:
            raise NotImplementedError('Need to define optimizer.')
        else:
            if isinstance(optimizer, str):
                assert optimizer.upper() in (
                    "ADAM", "RMSPROP", "SGD"
                )
                logging.info("Use optimizer: {}".format(optimizer))
                if optimizer.upper() == "ADAM":
                    return optim.Adam
                elif optimizer.upper() == "SGD":
                    return optim.SGD

    def __update_optimizer_learning_rate(self):
        pass

    def fit(self,
            train_loader,
            test_loader,
            epoch,
            loss_function=None,
            optimizer=None,
            reg_l1=False,
            reg_l2=True,
            learning_rate=0.001, \
            learning_rate_decay_rate=0.8,
            early_stopping_rounds=3,
            clip=-1,
            use_cuda=True,
            verbose=1,
            random_seed=7,
            learning_rates=None,
            save_checkpoints=True,
            reduce_loss=True,
            **kwargs):

        # Define model checkpoints setup:
        if self.checkpoint_dir is not None:
            if not os.path.isdir(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)
            checkpoint_path = os.path.join(self.checkpoint_dir, "PyTorch_training_checkpoint_{}".format(self.tic))
        logging.info("Set model checkpoint to {}.".format(checkpoint_path))

        logging.info("Start training PyTorch model.\nNeural network architecture: {}".format(self.model))

        # Pre-check CUDA status and assign device for training task:
        if use_cuda:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                logging.info("Will use CUDA for training.")
            else:
                _err_msg = "CUDA is not available!"
                logging.error(_err_msg); print(_err_msg)
                raise
        else:
            _warning_msg = "Warning: CUDA is not in use. CUDA availability: {}".format(torch.cuda.is_available())
            print(_warning_msg); logging.warning(_warning_msg)
            self.device = torch.device("cpu")

        # Assign random seed:
        assert isinstance(random_seed, int) and random_seed > 0, \
            TypeError("Input parameter random_seed can only be positive integer.")
        logging.info("Set random seed to {}.".format(random_seed))
        if torch.cuda.is_available() and use_cuda:
            torch.cuda.manual_seed_all(random_seed)
        elif not torch.cuda.is_available():
            print("Warning: CUDA is not available. Set random seed to CPU.")
            torch.manual_seed(random_seed)

        try:
            # Define placeholders for recording training processes:
            tra_loss_by_ep, val_loss_by_ep = {}, {} # Leave for later.
            # ep_train_loss = 0.0

            # Define placeholders for counting early stopping:
            _tolerance = 0
            _best_epoch = 1
            _best_loss = np.inf
            _best_model = None # May cause larger memory usage?
            _batch_train_loss = 0.0
            _total_train_loss = 0.0

            # Define learning rates:
            if learning_rates is not None:
                assert isinstance(learning_rates, list), \
                    TypeError("If learning_rates is not None, it must be list then.")
                adjustable_learning_rate = learning_rates[0]
                self.__use_learning_rate_decay = False # Define a flag of learning rate mode.
            elif learning_rates is None and isinstance(learning_rate_decay_rate, float):
                adjustable_learning_rate = learning_rate
                self.__use_learning_rate_decay = True
            else:
                raise ValueError("Double check learning rate setup.")

            # Training model:
            self.model = self.model.to(self.device)
            self.model.train()

            # Adjust trainable parameters:
            if hasattr(self.model, "trainable_params"):
                trainable_params = self.model.trainable_params()
                logging.info("Will only train partial parameters under trainable_params method.")
            else:
                trainable_params = self.model.parameters()
                logging.info("Will train all the parameters inherited from nn.Module.parameters method.")

            _loss_function = self.set_loss_function(loss_function=loss_function)

            # Initialize optimizer:
            _optimizer = self.set_optimizer(optimizer=optimizer)
            _optimizer = _optimizer(trainable_params, lr=adjustable_learning_rate)

            # Truncated. For future version: one data loader per model:
            # Adjustment for images augmentation:
            # if not isinstance(train_loader, list):
            #     train_loader_list = [train_loader]
            # else:
            #     train_loader_list = train_loader

            # Epoch loop:
            for ep in range(1, epoch+1):
                # Tic time for counting running time per epoch:
                _ep_tic = time.time()

                # Set up regularization placeholder:
                # reg_l1_loss, reg_l2_loss = None, None
                reg_l2_loss = 0 # torch.autograd.Variable(torch.Tensor(1), requires_grad=True)

                if self.__use_learning_rate_decay:
                    adjustable_learning_rate *= learning_rate_decay_rate ** (ep-1)
                else:
                    try:
                        adjustable_learning_rate = learning_rates[ep-1] # ep starts from 1.
                    except:
                        logging.warning("Reach the end of user-defined learning rates list. Will use last learning rate.")
                        pass

                # logging.info("Initial learning rate: {}".format(adjustable_learning_rate))
                # Update optimizer:
                for param_group_ in _optimizer.param_groups:
                    param_group_["lr"] = adjustable_learning_rate
                    logging.info("Epoch {}, changed learning rate to {}".format(ep, adjustable_learning_rate))

                # Truncated:
                # for _train_loader_idx, _train_loader in enumerate(train_loader_list):

                # Session started with train loader:
                _separator = '-'*35 + '\tEpoch: {}\t'.format(ep) + '-'*35
                print(_separator); logging.info(_separator)

                # Loop through batches:
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    _optimizer.zero_grad()
                    _output = self.model(data)
                    _loss = _loss_function(_output, target)

                    # Add regularization terms: To-do: add reg_l1?
                    if reg_l2:
                        # logging.info("Use L2 regularization for all trainable parameters.")
                        for _param in trainable_params:
                            reg_l2_loss += 0.5 * _param.norm(2)

                        _loss += reg_l2_loss
                    # Back-propagation:
                    _loss.backward()

                    # Clip the grad norm:
                    if clip > 0:
                        torch.nn.utils.clip_grad_norm(trainable_params, clip)
                    _optimizer.step()
                    # _batch_train_loss += _loss.item() # Monitor batch training error.
                    _total_train_loss += _loss.item()

                    # Verbose:
                    if ((batch_idx + 1) % verbose == 0):
                        _msg = 'Epoch: {epoch} Data Loader [{feed_data}/{whole_data} ({percentage:.0f}%)]\t Avg Train Loss per Batch: {avg_loss:.6f}'.format(
                            epoch=ep,
                            feed_data=(batch_idx + 1) * len(data), \
                            whole_data=len(train_loader.dataset), \
                            percentage=100 * (batch_idx + 1) / len(train_loader), \
                            avg_loss=_total_train_loss / (batch_idx + 1))
                        print(_msg); logging.info(_msg)

                # Reset placeholder of epoch training loss:
                tra_loss_by_ep[ep] = _total_train_loss
                _total_train_loss = 0.0

                # Calculate score from evaluation data set:
                test_loss = self.eval(self.model, test_loader)
                val_loss_by_ep[ep] = test_loss
                logging.info("Epoch {}, Test Loss: {:.6f}".format(ep, test_loss))

                # Show error per epoch:
                # if ep % verbose == 0:
                # print("Training Epoch: {}\tAvg Loss: {:.6f}".format(ep, avg_epoch_loss))

                if test_loss < _best_loss:
                    _best_loss = test_loss
                    _best_epoch = ep
                    _best_model = self.model
                    _tolerance = 1
                    # Save model checkpoint:
                    if save_checkpoints:
                        try:
                            torch.save(_best_model, checkpoint_path)
                            _msg = "Saved model to checkpoint."
                            logging.info(_msg); print(_msg)
                        except Exception as e:
                            _msg = "Failed in saving trained model as checkpoint. Error: {}".format(e)
                            logging.error(_msg); print(_msg)
                            raise
                else:
                    if _tolerance < early_stopping_rounds:
                        _tolerance += 1
                        _msg = "Loss did not improve. Set tolerance to {}.".format(_tolerance)
                        logging.info(_msg); print(_msg)
                    # Truncated early stopping, updated it to learning rate decay:
                    else:
                        _msg = "Reach maximum tolerance. Early stopped."
                        logging.info(_msg); print(_msg)
                        break

                # Reset epoch loss:
                ep_train_loss = 0.0

                # Count epoch running time:
                _ep_time_cost = (time.time() - _ep_tic) / 60.0
                _msg = "Running time for this epoch: {:.2f} minutes.".format(_ep_time_cost)
                logging.info(_msg); print(_msg)

            # Log best scores:
            _msg = "Best training epoch: {}, best training loss score: {:.6f}".format(_best_epoch, _best_loss)
            logging.info(_msg); print(_msg)

            # Replace class model with best trained model:
            self.model = _best_model

            return _best_model
        except Exception as e:
            print("Error during training PyTorch model: {}".format(e))
            raise

    def eval(self, model, test_loader, pred_mode="binary", eval_func=None):
        # raise NotImplementError("Sub-class needs to define the eval function.") # Debug.

        try:
            model.eval()
            test_loss = 0.0

            # Placeholders for calculating accuracy:
            truth = []
            proba = []

            # Truncated:
            # if not isinstance(test_loaders, list):
            #     test_loader_list = [test_loaders]
            # else:
            #     test_loader_list = test_loaders

            # Loop through test data loaders:
            # for test_loader_idx, test_loader in enumerate(test_loader_list):

            # Loop through data batch per loader:
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                test_loss += torch.nn.functional.cross_entropy(output, target).item()

                # Binary classification, calculate accuracy:
                # proba_, bin_pred_ = output.max(dim=1)

                # Tweak this by models:
                if pred_mode == 'binary':
                    proba_ = output[:, 1]
                else:
                    proba_ = output[:, 1]

                proba += proba_.cpu().detach().numpy().reshape(-1, ).tolist()
                truth += target.cpu().numpy().reshape(-1, ).tolist()

            # Calculate total metrics:
            test_loss /= len(test_loader)
            eval_logloss = log_loss(truth, np.exp(proba))
            pred_class = [1 if ele>=0.5 else 0 for ele in np.exp(proba)]
            eval_accuracy = accuracy_score(truth, pred_class)

            print("\nTest set:\nLog Loss: {:.6f}\tAccuracy: {:.4f}%\n".format(eval_logloss, eval_accuracy*100))
            return eval_logloss
        except Exception as e:
            print("Failed in evaluate the model. Error: {}".format(e))

    # Need to be updated for sub-class instances:
    def __model_inference(self, prediction_loader, model):

        # Switch model to eval mode:
        model.eval()

        # Create a place holder to catch probability predictions:
        batch_prediction = []
        for i, (data, target) in tqdm(enumerate(prediction_loader), total=len(prediction_loader)):
            # if i>0:
            #     break

            data, target = data.cuda(), target.cuda()
            _prediction = model(data)
            batch_prediction += _prediction[:, 1].cpu().detach().numpy().tolist()
            gc.collect()

        return batch_prediction

    # Need to be updated for sub-class instances:
    def predict_proba(self, model, pred_data_dir, image_size, normalize):

        # Switch to eval mode:
        model.eval()

        # Make placeholder for ensemble results:
        predictions = dict()

        # Make tta processes to do augmented prediction:
        tta_preprocess = [preprocess(normalize, image_size), preprocess_hflip(normalize, image_size)]

        tta_preprocess += make_transforms([transforms.Resize((image_size + 20, image_size + 20))],
                                          [transforms.ToTensor(), normalize],
                                          five_crops(image_size))

        tta_preprocess += make_transforms([transforms.Resize((image_size + 20, image_size + 20))],
                                          [HorizontalFlip(), transforms.ToTensor(), normalize],
                                          five_crops(image_size))

        logging.info("TTA processes length: {}".format(len(tta_preprocess)))

        print("Start ensemble-prediction:\n")
        for _idx, _tta_transform in tqdm(enumerate(tta_preprocess), total=len(tta_preprocess)):
            prediction_dataset = test_dataset(directory=pred_data_dir, transforms=_tta_transform, prediction_mode=True)
            prediction_dataloader = DataLoader(dataset=prediction_dataset, batch_size=32, shuffle=False)

            logging.info("Starting {} tta transform...".format(_idx))
            _tic = time.time()
            _batch_prediction = self.__model_inference(prediction_loader=prediction_dataloader, model=model)
            predictions["tta_{}".format(_idx)] = _batch_prediction
            _toc = time.time()
            _time_cost_in_min = (_toc - _tic) / 60
            logging.info("Finished. Time cost: {:.4f}".format(_time_cost_in_min))

            del prediction_dataset, prediction_dataloader
            gc.collect()

        return predictions



































