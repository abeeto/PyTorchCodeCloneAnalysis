import numpy
import argparse
import logging

import torch

from myTorch import Experiment
from myTorch.memnets.recurrent_net import Recurrent
from myTorch.task.ssmnist_task import SSMNISTData
from myTorch.task.mnist_task import PMNISTData
from myTorch.utils.logger import Logger
from myTorch.utils import MyContainer, get_optimizer, create_config
import torch.nn.functional as F
from torch.autograd import grad

parser = argparse.ArgumentParser(description="Algorithm Learning Task")
parser.add_argument("--config", type=str, default="config/default.yaml", help="config file path.")
parser.add_argument("--force_restart", type=bool, default=False, help="if True start training from scratch.")


def get_data_iterator(config):
    if config.task == "ssmnist":
        data_iterator = SSMNISTData(config.data_folder, num_digits=config.num_digits,
                                    batch_size=config.batch_size, seed=config.seed)
    elif config.task == "pmnist":
        data_iterator = PMNISTData(batch_size=config.batch_size, seed=config.seed)

    return data_iterator


def evaluate(experiment, model, config, data_iterator, tr, logger, device, tag):

    logging.info("Doing {} evaluation".format(tag))

    correct = 0.0
    num_examples = 0.0

    while True:

        data = data_iterator.next(tag)

        if data is None:
            break

        model.reset_hidden(batch_size=config.batch_size)

        accuracy = torch.zeros(config.batch_size).to(device)
        num_outputs = torch.zeros(config.batch_size).to(device)

        for i in range(0, data["datalen"]):

            x = torch.from_numpy(numpy.asarray(data['x'][i])).to(device)
            y = torch.from_numpy(numpy.asarray(data['y'][i])).to(device)
            mask = torch.from_numpy(numpy.asarray(data['mask'][i])).to(device)

            output = model(x)

            values, indices = torch.max(output, 1)


            accuracy += (indices == y).to(device, dtype=torch.float32) * mask
            num_outputs += mask


        correct += (accuracy == num_outputs).sum()
        num_examples += len(data['x'][0])


    final_accuracy = correct.item() / num_examples
    logging.info(" epoch {}, {} accuracy: {}".format(tr.epochs_done, tag, final_accuracy))

    if config.use_tflogger:
        logger.log_scalar("{}_accuracy".format(tag), final_accuracy, tr.epochs_done)


def train(experiment, model, config, data_iterator, tr, logger, device):
    """Training loop.

    Args:
        experiment: experiment object.
        model: model object.
        config: config dictionary.
        data_iterator: data iterator object
        tr: training statistics dictionary.
        logger: logger object.
    """

    for step in range(tr.updates_done, config.max_steps):

        if tr.updates_done == 0:
            experiment.save("initial")
        if config.inter_saving is not None:
            if tr.updates_done in config.inter_saving:
                experiment.save(str(tr.updates_done))

        data = data_iterator.next("train")

        if data is None:

            tr.epochs_done += 1
            evaluate(experiment, model, config, data_iterator, tr, logger, device, "valid")
            evaluate(experiment, model, config, data_iterator, tr, logger, device, "test")
            data_iterator.reset_iterator()
            data = data_iterator.next("train")

        seqloss = 0

        model.reset_hidden(batch_size=config.batch_size)

        for i in range(0, data["datalen"]):

            x = torch.from_numpy(numpy.asarray(data['x'][i])).to(device)
            y = torch.from_numpy(numpy.asarray(data['y'][i])).to(device)
            mask = torch.from_numpy(numpy.asarray(data['mask'][i])).to(device)

            model.optimizer.zero_grad()

            output = model(x)

            loss = F.cross_entropy(output, y, reduce=False)

            loss = (loss * mask).sum()


            seqloss += loss

        seqloss /= float(data["mask"].sum())
        tr.ce["train"].append(seqloss.item())
        running_average = sum(tr.ce["train"]) / len(tr.ce["train"])

        if config.use_tflogger:
            logger.log_scalar("running_avg_loss", running_average, step + 1)
            logger.log_scalar("train loss", tr.ce["train"][-1], step + 1)

        seqloss.backward(retain_graph=False)


        total_norm = torch.nn.utils.clip_grad_norm(model.parameters(), config.grad_clip_norm)
        tr.grad_norm.append(total_norm)

        if config.use_tflogger:
            logger.log_scalar("inst_total_norm", total_norm, step + 1)

        model.optimizer.step()

        tr.updates_done += 1

        if tr.updates_done % 1 == 0:
            logging.info("examples seen: {}, inst loss: {}".format(tr.updates_done * config.batch_size,
                                                                   tr.ce["train"][-1]))

        if tr.updates_done % config.save_every_n == 0:
            experiment.save()


def create_experiment(config):
    """Creates an experiment based on config."""

    device = torch.device(config.device)
    logging.info("using {}".format(config.device))

    experiment = Experiment(config.name, config.save_dir)
    logger = None
    if config.use_tflogger:
        logger = Logger(config.tflog_dir)

    torch.manual_seed(config.rseed)

    model = Recurrent(device, config.input_size, config.output_size,
                      num_layers=config.num_layers, layer_size=config.layer_size,
                      cell_name=config.model, activation=config.activation,
                      output_activation="linear", layer_norm=config.layer_norm,
                      identity_init=config.identity_init, chrono_init=config.chrono_init,
                      t_max=config.t_max, use_relu=config.use_relu, memory_size=config.memory_size,
                      k=config.k, phi_size=config.phi_size, r_size=config.r_size).to(device)

    data_iterator = get_data_iterator(config)

    optimizer = get_optimizer(model.parameters(), config)
    model.register_optimizer(optimizer)

    tr = MyContainer()
    tr.updates_done = 0
    tr.epochs_done = 0
    tr.ce = {}
    tr.ce["train"] = []
    tr.accuracy = {}
    tr.accuracy["valid"] = []
    tr.accuracy["test"] = []
    tr.grad_norm = []


    experiment.register_experiment(model=model, config=config, logger=logger, train_statistics=tr,
        data_iterator=data_iterator)

    return experiment, model, data_iterator, tr, logger, device

def evolve_and_QR(Q,J):
    Z = np.dot(J.data,Q)
    q,r = np.linalg.qr(Z,mode='reduced')
    return q, np.diag(r)
    #s = np.diag(np.sign(np.diag(r))) #changes QR sign convention to that used by tensorflow 
    #return q.dot(s), np.diag(r.dot(s))

def calculate_LS(model, data_iterator, input_flag, device, num_exp):
    """Lyapunov spectrum calculation using QR decomposition"""

    data_iterator.reset_iterator()
    data = data_iterator.next(input_flag)
    
    #store hidden states
    h_list = list()
    model.reset_hidden(batch_size=1)
    h_list.append(model._h_prev[0]["h"])
    h_list[-1].requires_grad_()
    for i in range(0, data["datalen"]):

        x = torch.from_numpy(numpy.asarray(data['x'][i])).to(device)
        y = torch.from_numpy(numpy.asarray(data['y'][i])).to(device)
        mask = torch.from_numpy(numpy.asarray(data['mask'][i])).to(device)
        
        x = x[0].unsqueeze(1)

        output = model(x)
        h_list.append(model._h_prev[0]["h"])
        h_list[-1].requires_grad_()

    #store Jacobians
    j_store = list()
    for j in range(0, data["datalen"]):
        for i in range(1, len(h_list)):
            model.optimizer.zero_grad()
            j_rows = list()
            for k in range(0, len(h_list[i][0])):
                j_list.append(grad(h_list[i][0][k], h_list[i-1], retain_graph=True)[0])
            j_store.append(torch.vstack(j_list))

    #compute LS
    state_dim = model._h_prev[0]["h"].shape[1]
    #state_dim=sum([var.shape[1] for var in model._h_prev[0].values])  #generalizes to models with multiple state variables
    Q = torch.eye(state_dim).to(device) #would random vectors be better here?
    Q = Q[:,:num_exp]
    Q = Q.cuda()
    Q = Q.unsqueeze(0)
    r_diag_store = []
    for i in range(len(j_store)):
        Q, r_diag_vec = evolve_and_QR(Q,j_store[i])
        r_diag_store.append(r_diag_vec)
    LEs = np.sum(np.log2(np.vstack(r_diag_store)),axis=0)/len(j_store)
    
    return LEs

    # if config.use_tflogger:
    #     logger.log_scalar("running_avg_loss", running_average, step + 1)
    #     logger.log_scalar("train loss", tr.ce["train"][-1], step + 1)

    #if config.use_tflogger:
        #logger.log_scalar("inst_total_norm", total_norm, step + 1)

    #if tr.updates_done % 1 == 0:
        #logging.info("examples seen: {}, inst loss: {}".format(tr.updates_done * config.batch_size,
                                                               #tr.ce["train"][-1]))

    #if tr.updates_done % config.save_every_n == 0:
        #experiment.save()

def analyze_dynamics(model, data_iterator, device, num_exp=10):
    """Runs Lyapunov spectrum calculation over different input types"""
    
    LEs_store=dict()
    for input_flag in ('test','train','zeros'):
        LEs=calculate_LEs(model, data_iterator, input_flag,device, num_exp)
        LEs_store[input_flag]=LEs
    
    return LEs_store

def run_experiment(args):
    """Runs the experiment."""

    config = create_config(args.config)

    logging.info(config.get())

    experiment, model, data_iterator, tr, logger, device = create_experiment(config)

    if not args.force_restart:
        if experiment.is_resumable():
            experiment.resume()
    else:
        experiment.force_restart()

    #train(experiment, model, config, data_iterator, tr, logger, device)
    analyze_dynamics(model, data_iterator, device)


if __name__ == '__main__':
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    run_experiment(args)
