from flask import Flask, send_file, request, Blueprint
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import numpy as np
from itertools import cycle
from typing import List, Union
import requests
import socket
from contextlib import closing
from io import BytesIO
from uuid import uuid4


# possibly silly, stolen from 
# https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number
def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


class QueuedDataLoader:
    def __init__(self, dataloader, name, queue_size, app):
        self.dataloader = cycle(dataloader)
        print(f'Filling up queue for dataloader {name}.')
        self.queue = self.fill_queue(queue_size)
        self.log = [[] for _ in range(queue_size)]
        self.dl_len = str(len(dataloader))
        
        bp = Blueprint(name, __name__)
        bp.route(f'/get_batch', methods=['GET'])(self.send_batch)
        bp.route(f'/get_len', methods=['GET'])(self.get_len)
        app.register_blueprint(bp, url_prefix=f'/data/{name}')
        
    def get_len(self):
        return str(self.dl_len)
        
    def fill_queue(self, size):
        return [batch for batch, _ in zip(self.dataloader, range(size))]
    
    def refill_step(self):
        """
        this is not done:
        - need to give out most-used batches
        - need to refill intelligently / only sometimes
        """
        counts = [len(l) for l in self.log]
        if max(counts) == 0:
            return
        replace_index = np.argmax(counts)
        print(self.log)
        print(f'Refilling {replace_index}')
        self.queue[replace_index] = next(self.dataloader)
        self.log[replace_index] = []
        
    def _get_batch(self, client_id):
        while True:
            for batch, previous_clients in zip(self.queue, self.log):
                if client_id not in previous_clients:
                    previous_clients.append(client_id)
                    self.refill_step()
                    return batch
            
    def send_batch(self):
        client_id = request.args['client_id']
        batch = self._get_batch(client_id)
        b = BytesIO()
        torch.save(batch, b)
        b.seek(0)
        return send_file(b, download_name='batch.pt')
        

def serve_dataloaders(
    dataloaders: List[DataLoader],
    names: List[str],
    queue_size: Union[int, List[int]],
    port=None,
):
    assert len(dataloaders) == len(names), 'names and dataloaders must have the same length.'
    assert len(set(names)) == len(names), 'names must be unique.'
    if isinstance(queue_size, list):
        assert len(queue_size) == len(dataloaders), (
            'When passing a list of queue sizes, '
            'it must be the same length as dataloaders.'
        )
    app = Flask(__name__)
    port = port or find_free_port()
    
    if isinstance(queue_size, int):
        queue_size = [queue_size] * len(dataloaders)
        
    for dataloader, name, size in zip(dataloaders, names, queue_size):
        QueuedDataLoader(dataloader, name, size, app)
        
    app.route('/available', methods=['GET'])(lambda: names)
    app.run(port=port)
    
    
class ClientDataLoader:
    def __init__(self, name, port, client_id):
        self.name = name
        self.port = port
        self.id = client_id
        self.length = self._get_len()
        
    def get_batch(self):
        r = requests.get(
            f'http://127.0.0.1:{self.port}/data/{self.name}/get_batch',
            {'client_id': self.id}
        )
        b = BytesIO(r.content)
        return torch.load(b)
        
    def _get_len(self):
        r = requests.get(f'http://127.0.0.1:{self.port}/data/{self.name}/get_len')
        return int(r.text)
        
    def __len__(self):
        return self.length
    
    def __iter__(self):
        for _ in range(len(self)):
            yield self.get_batch()
    
    
def get_clients(names, port):
    server_available = requests.get(f'http://127.0.0.1:{port}/available').json()
    assert set(names).issubset(server_available), (
        'Some of the names are not available on the server. '
        f'Available names: {", ".join(server_available)}. '
        f'Requested names: {", ".join(names)}.'
    )
    training_id = str(uuid4())
    return [ClientDataLoader(name, port, training_id) for name in names]
    