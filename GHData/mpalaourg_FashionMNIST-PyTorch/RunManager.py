import pandas as pd
import time
import json
from collections import OrderedDict


class RunManager():
    def __init__(self):
        """ Class constructor """
        self.epoch_count = 0
        self.epoch_loss = 0
        self.epoch_num_correct = 0
        self.epoch_start_time = None

        self.run_params = None
        self.run_count = 0
        self.run_data = []
        self.run_start_time = None

        self.network = None
        self.loader = None

    def begin_run(self, run, network, loader):
        """ Function to initialize each individual run """
        self.run_start_time = time.time()    # start time of the current run

        self.run_params = run                # save the current run parameters
        self.run_count += 1                  # increment the current run by one

        self.network = network               # save our network
        self.loader = loader                 # save our dataloader

    def end_run(self):
        """ Function to wrap up the current run """
        self.epoch_count = 0                 # restart the epoch count
        print(f"Done with run {self.run_count}")

    def begin_epoch(self):
        """ Function to initialize each individual epoch of each run"""
        self.epoch_start_time = time.time()  # start time of the current epoch

        self.epoch_count += 1                # increment current epoch by one
        self.epoch_loss = 0                  # zero current loss
        self.epoch_num_correct = 0           # zero current number of correct predictions

    def end_epoch(self):
        """ Function to wrap up the current epoch"""
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        loss = self.epoch_loss / len(self.loader.dataset)
        accuracy = self.epoch_num_correct / len(self.loader.dataset)

        # Track training loop perfomance #
        results = OrderedDict()
        results["run"] = self.run_count
        results["epoch"] = self.epoch_count
        results['loss'] = loss
        results["accuracy"] = accuracy
        results['epoch duration'] = epoch_duration
        results['run duration'] = run_duration
        for k, v in self.run_params._asdict().items():
            results[k] = v
        self.run_data.append(results)

    def track_loss(self, loss, batch):
        """ Function to track the loss of each batch of images """
        self.epoch_loss += loss.item() * batch[0].shape[0]

    def track_num_correct(self, preds, labels):
        """ Function to track the number of correct predictions of each batch of images """
        self.epoch_num_correct += self._get_num_correct(preds, labels)

    def _get_num_correct(self, preds, labels):
        """ Function to calculate the number of correct predictions of each batch of images """
        return preds.argmax(dim=1).eq(labels).sum().item()

    def save(self, fileName):
        """ Function to save the results in JSON and .csv format for each training loop"""
        pd.DataFrame.from_dict(
            self.run_data, orient='columns'
        ).to_csv(f'{fileName}.csv')

        with open(f'{fileName}.json', 'w', encoding='utf-8') as f:
            json.dump(self.run_data, f, ensure_ascii=False, indent=4)
