import torch
import torch.distributed.rpc as rpc
from client import client
import logging
from copy import deepcopy

class server(object):
    def __init__(self,
                 model,
                 rank,
                 world_size,
                 args=None):

        self.create_logger(rank)

        self.n_clients = world_size - 1
        self.model = model
        self.client_rrefs = []
        self.rank = rank
        self.world_size = world_size
        self.initialize_client_modules(args)

        self.logger.info(f"Server {self.rank} Initialized")

    def initialize_client_modules(self,args):
        self.logger.info(f"Initialize {self.world_size-1} Clients")
        for rank in range(self.world_size-1):
            self.client_rrefs.append(
                                    rpc.remote(f"worker{rank+1}",
                                           client,
                                           args=(self.model,rank+1,self.world_size,args))
                                )

    def send_global_model(self):
       self.logger.info("Sending Global Parameters")
       check_global = [client_rref.remote().load_global_model(deepcopy(self.model.state_dict())) for client_rref in self.client_rrefs]
       for check in check_global:
           check.to_here()

    def train(self):
        self.logger.info("Initializing Trainig")
        check_train = [client_rref.remote(timeout=0).train() for client_rref in self.client_rrefs]
        for check in check_train:
            check.to_here(timeout=0)

    def evaluate(self):
        self.logger.info("Initializing Evaluation")
        total = []
        num_corr = []
        check_eval = [client_rref.remote(timeout=0).evaluate() for client_rref in self.client_rrefs]
        for check in check_eval:
            corr,tot = check.to_here(timeout=0)
            total.append(tot)
            num_corr.append(corr)

        self.logger.info("Accuracy over all data: {:.3f}".format(sum(num_corr)/sum(total)))

    def aggregate(self):
        self.logger.info("Aggregating Models")
        check_n_sample = [client_rref.rpc_async().send_num_train() for client_rref in self.client_rrefs]
        n_samples = [check.wait() for check in check_n_sample]
        n_total = sum(n_samples)

        check_params = [client_rref.rpc_async().send_local_model() for client_rref in self.client_rrefs]
        client_params = [check.wait() for check in check_params]

        global_model_state_dict = deepcopy(self.model.state_dict())

        for name,param in self.model.named_parameters():
            global_model_state_dict[name] = torch.zeros_like(global_model_state_dict[name])
            for n_train,client_param in zip(n_samples,client_params):
                global_model_state_dict[name] = global_model_state_dict[name] + n_train/n_total * client_param[name]

        self.model.load_state_dict(global_model_state_dict)

    def update_clients(self):
        self.aggregate()
        self.send_global_model()

    def create_logger(self,rank):
        self.logger = logging.getLogger(f'server{rank}')
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.FileHandler(f"server{rank}.log",mode='w',encoding='utf-8'))
