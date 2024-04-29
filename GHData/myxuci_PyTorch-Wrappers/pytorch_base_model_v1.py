# Dev Note:
# Log Finished.
# early stop
# Autosave
# Learning rate decay

import io
import os
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms

class PyTorchBaseModel(object):

    def __init__(self, model=None, loss_function=None, optimizer=None, log_dir=None, checkpoint_dir=None):

        # Tic:
        self.tic = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        # Initialize log:
        self.__init_logging(log_dir=log_dir)

        # Define model checkpoints:
        self.checkpoint_dir = checkpoint_dir

        # Define model:
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
            self.__log_stream = io.StringIO()
            logging.basicConfig(
                stream=self.__log_stream,
                level=logging.INFO,
                format="[%(asctime)s (Local Time)] %(levelname)s : %(message)s",
                # Local time may vary for cloud services.
                datefmt="%m/%d/%Y %I:%M:%S %p"
            )
        else:
            self.__log_stream = None
            if not os.path.isdir(log_dir):
                os.makedirs(log_dir)

            # date_str = datetime.now().strftime('%Y-%m-%d_%H-%M')
            log_file = 'PyTorch_log_{}.txt'.format(self.tic)

            # reload(logging)  # bad
            logging.basicConfig(
                filename=os.path.join(log_dir, log_file),
                level=logging.INFO,
                format="[%(asctime)s (Local Time)] %(levelname)s : %(message)s",
                datefmt='%m/%d/%Y %I:%M:%S %p'
            )
            # logging.getLogger().addHandler(logging.StreamHandler())
        logging.info("Created log at {}.".format(self.tic))

    def get_log(self):
        if self.__log_stream is not None:
            return self.__log_stream.get_value()
        else:
            print("Log stream does not exist.")
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
                    return F.nll_loss
                if loss_function.upper() == "BINARY_CROSS_ENTROPY":
                    return F.binary_cross_entropy
                if loss_function.upper() == "BINARY_CROSS_ENTROPY_WITH_LOGITS":
                    return F.binary_cross_entropy_with_logits
                if loss_function.upper() == "CROSS_ENTROPY":
                    return F.cross_entropy
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

    def fit(self, train_loader, epoch, loss_function=None, optimizer=None, learning_rate=0.01, \
            clip=-1, use_cuda=True, early_stopping=5, verbose=1, random_seed=7, **kwargs):
        '''
        :param train_loader:
        :param epoch:
        :param use_cuda:
        :param early_stopping:
        :param verbose:
        :param kwargs: Used for adjusting the parameters of loss functions and optimizers.
        :return:
        '''
        logging.info("Training PyTorch model.")
        logging.info("Neural network architecture: {}".format(self.model.eval()))

        if isinstance(random_seed, int) and random_seed > 0:
            torch.manual_seed(random_seed)
        else:
            raise TypeError("Input parameter random_seed can only be positive integer.")

        if use_cuda:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                print("CUDA is not available!")
                raise
        else:
            print("Warning: CUDA is not in use. CUDA availability: {}".format(torch.cuda.is_available()))
            device = torch.device("cpu")

        try:
            # Define placeholders for recording training processes:
            train_loss_by_ep = dict() # Leave for later.
            # ep_train_loss = 0.0

            # Define model checkpoints setup:
            if self.checkpoint_dir is not None:
                save_checkpoints = True
                if not os.path.isdir(self.checkpoint_dir):
                    os.makedirs(self.checkpoint_dir)
                checkpoint_path = os.path.join(self.checkpoint_dir, "PyTorch_training_checkpoint_{}".format(self.tic))
            else:
                save_checkpoints = False

            # Define placeholders for counting early stopping:
            _tolerance = 0
            _best_epoch = 1
            _best_loss = 999999.99
            _best_model = None # May cause larger memory usage?
            _train_loss = 0.0

            # Training model:
            self.model = self.model.to(device)
            self.model.train()
            _loss_function = self.set_loss_function(loss_function=loss_function)
            _optimizer = self.set_optimizer(optimizer=optimizer)
            _optimizer = _optimizer(self.model.parameters(), lr=learning_rate)

            # Need to have a generator for feeding data.
            for ep in range(1, epoch+1):
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(device), target.to(device)
                    _optimizer.zero_grad()
                    _output = self.model(data)
                    _loss = _loss_function(_output, target)
                    _loss.backward()

                    # Clip the grad norm:
                    if clip > 0:
                        torch.nn.utils.clip_grad_norm(self.model.parameters(), clip)
                    _optimizer.step()
                    _train_loss += _loss
                    if batch_idx % verbose == 0:
                        print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(ep, batch_idx * len(data), \
                                                                    len(train_loader.dataset), \
                                                                    100. * batch_idx / len(train_loader), \
                                                                    _train_loss.data[0]/verbose))
                        # Reset placeholder of batch training loss:
                        _train_loss = 0.0
                
                # This part needs second thoughts...
                # avg_epoch_loss = ep_train_loss.data[0]/(batch_idx+1)
                avg_epoch_loss = _loss.item() / batch_idx
                logging.info("Epoch {}, Average Loss: {:.6f}".format(ep, avg_epoch_loss))
                
                # Show error per epoch:
                # if ep % verbose == 0:
                    # print("Training Epoch: {}\tAvg Loss: {:.6f}".format(ep, avg_epoch_loss))

                if avg_epoch_loss <= _best_loss:
                    _best_loss = avg_epoch_loss
                    _best_epoch = ep
                    _best_model = self.model
                    _tolerance = 1
                else:
                    if _tolerance <= early_stopping:
                        _tolerance += 1
                    else:
                        print("Early stopped.")
                        logging.info("Early stopped.")
                        break

                # Reset epoch loss:
                ep_train_loss = 0.0

            # Log best scores:
            logging.info("Best training epoch: {}, best training loss score: {:.6f}".format(_best_epoch, _best_loss))

            # Replace class model with best trained model:
            self.model = _best_model

            # Save model checkpoint:
            if save_checkpoints:
                try:
                    torch.save(_best_model, checkpoint_path)
                except Exception as e:
                    print("Failed in saving trained model as checkpoint. Error: {}".format(e))
                    logging.error("Failed in saving trained model as checkpoint. Error: {}".format(e))
                    raise

            return _best_model
        except Exception as e:
            print("Error during training PyTorch model: {}".format(e))
            raise

    def predict(self):
        pass
































