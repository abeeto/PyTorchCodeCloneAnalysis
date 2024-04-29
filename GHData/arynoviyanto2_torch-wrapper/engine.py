# Copyright Ary Noviyanto 2021

from tqdm import tqdm
from torch.autograd import Variable

from helpers import MeasureProgression, save_as_model, save_as_best_model
from performance_evaluation import PerformanceEvaluation, Metric
import numpy as np

class Engine:

    def __init__(self, ml_model, params):
        self.ml_model = ml_model
        self.params = params

    def run_training(self, train_dataloader, test_dataloader):
        best_performance_value = 0

        # For each epoch
        for epoch in range(self.params['epochs']):
            print("Epoch {}/{}".format(epoch, self.params['epochs']))

            self.train(train_dataloader)
            pe = self.evaluate(test_dataloader)

            metrics = pe.get_performance_metrics()
            accuracy = metrics[Metric.ACCURACY]
            is_best = accuracy > best_performance_value

            model_name = self.ml_model.manifest().get_name()
            model_state = {
                'state_dict': self.ml_model.manifest().state_dict(),
                'optim_dict': self.ml_model.optimizer().state_dict()
            }

            if is_best:
                save_as_best_model(model_name, model_state, self.params['storage_dir'])
                best_performance_value = accuracy

            save_as_model(epoch, model_name, model_state, self.params['storage_dir'])

        return best_performance_value

    def train(self, train_dataloader):
        n = len(train_dataloader) # number of batches

        model = self.ml_model.manifest()
        model.train()

        loss_running_avg = MeasureProgression()

        # Train number of mini batches 
        with tqdm(total=n, desc="Train") as progress:
            for data_batch, label_batch in train_dataloader: # for each mini batch
                
                data_batch = Variable(data_batch)
                label_batch = Variable(label_batch)

                predicted_labels = model(data_batch)
                loss = self.ml_model.loss_func(predicted_labels, label_batch)

                self.ml_model.optimizer().zero_grad()
                loss.backward()
                self.ml_model.optimizer().step()

                loss_running_avg.add(loss.item())

                progress.set_postfix(loss='{avg_loss:.2f}'.format(avg_loss=loss_running_avg.getMeasure()))
                progress.update()

    def evaluate(self, test_dataloader, saved_model=None):
        n = len(test_dataloader)

        model = saved_model if saved_model is not None else self.ml_model.manifest()
        model.eval()

        loss_running_avg = MeasureProgression()

        predicted_labels_arr = None
        labels_arr = None

        # Test number of mini batches 
        with tqdm(total=n, desc="Eval") as progress:
            for data_batch, label_batch in test_dataloader: # for each mini batch
                
                data_batch = Variable(data_batch)
                label_batch = Variable(label_batch)

                predicted_labels = model(data_batch)
                loss = self.ml_model.loss_func(predicted_labels, label_batch)

                tmp_predicted_labels = predicted_labels.data.cpu().numpy() 
                tmp_labels = label_batch.data.cpu().numpy()
                predicted_labels_arr = tmp_predicted_labels if predicted_labels_arr is None else np.concatenate((predicted_labels_arr, tmp_predicted_labels))
                labels_arr = tmp_labels if labels_arr is None else np.concatenate((labels_arr, tmp_labels))

                loss_running_avg.add(loss.item())

                progress.set_postfix(loss='{avg_loss:.2f}'.format(avg_loss=loss_running_avg.getMeasure()))
                progress.update()

        return PerformanceEvaluation(predicted_labels_arr, labels_arr, model.num_targets)
