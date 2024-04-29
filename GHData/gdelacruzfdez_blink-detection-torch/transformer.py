import sys
import dataloader
import network
import numpy as np
import torch
import evaluator
from sklearn import metrics
from tqdm import tqdm
from PIL import Image
from functional import seq
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from evaluation import evaluate, evaluatePartialBlinks
from sklearn.utils.class_weight import compute_class_weight
from functools import reduce
from abc import ABC, abstractmethod
from torchinfo import summary
from numpy.lib.stride_tricks import sliding_window_view


class TransformerModel(ABC):
    TRAIN_TRANSFORM = transforms.Compose([
        transforms.Resize((64, 64), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    TEST_TRANSFORM = transforms.Compose([
        transforms.Resize((64, 64), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    def __init__(self, params, evaluator, cuda):
        self.params = params
        self.cuda = cuda
        self.evaluator = evaluator

        self.batch_size = params.get('batch_size')
        self.dims = params.get('dims')
        self.cnn_model_file = params.get('cnn_model_file')
        self.transformer_model_file = params.get('transformer_model_file')
        self.epochs = params.get('epochs')
        self.sequence_len = params.get('sequence_len')
        self.eval_mode = params.get('eval_mode')
        self.lr = params.get('lr')
        self.base_file_name = 'transformer'

        if 'train_dataset_dirs' in params:
            self.train_dataset_dirs = seq(params.get('train_dataset_dirs')) \
                .map(lambda x: "{}/{}".format(params.get('datasets_base_path'), x))
        if 'test_dataset_dirs' in params:
            self.test_dataset_dirs = seq(params.get('test_dataset_dirs')) \
                .map(lambda x: "{}/{}".format(params.get('datasets_base_path'), x))

        self.train_videos = None if not 'train_videos' in params else params.get('train_videos')
        self.test_videos = None if not 'test_videos' in params else params.get('test_videos')

        self.best_f1 = -1
        self.best_epoch = -1
        self.current_f1 = -1

        self.__initialize_cnn_model()
        self.initialize_transformer_model()
        self.__initialize_log_file()
        self.initialize_train_loader()
        self.initialize_evaluation_loader()
        self.__initialize_training_parameters()

    def __initialize_log_file(self):
        print('TODO: Transformer log file')
        # self.LOG_FILE_HEADER = 'EPOCH,TRAIN_LOSS,TRAIN_ACCURACY,TRAIN_PRECISION,TRAIN_RECALL,TRAIN_F1,EVAL_F1,EVAL_PRECISION,EVAL_RECALL,EVAL_TP,EVAL_FP,EVAL_TN,EVAL_FN'
        # separator = '-'
        # self.base_file_name = "LSTM_train[{}]_test[{}]_batch_[{}]_dims[{}]_lr[{}]_hidden-units[{}]_lstm-layers[{}]_sequence-len[{}]_epochs[{}]" \
        #     .format(separator.join(self.params.get('train_dataset_dirs')),
        #             separator.join(self.params.get('test_dataset_dirs')),
        #             self.batch_size,
        #             self.dims,
        #             self.lr,
        #             self.lstm_hidden_units,
        #             self.lstm_layers,
        #             self.sequence_len,
        #             self.epochs)
        #
        # self.log_file = open(self.base_file_name + '.log', 'w')
        #
        # self.log_file.write(self.base_file_name + '\n\n')
        # self.log_file.write(self.LOG_FILE_HEADER)

    @abstractmethod
    def initialize_train_loader(self):
        pass

    @abstractmethod
    def initialize_evaluation_loader(self):
        pass

    def __initialize_cnn_model(self):
        self.cnn_model = network.SiameseNetV2(self.dims)
        self.cnn_model.load_state_dict(torch.load(self.cnn_model_file))
        self.cnn_model.eval()
        if self.cuda:
            self.cnn_model = self.cnn_model.cuda()

    def initialize_transformer_model(self):
        emsize = 256  # embedding dimension
        d_hid = 1024  # dimension of the feedforward network model in nn.TransformerEncoder
        nlayers = 4  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        nhead = 2  # number of heads in nn.MultiheadAttention
        dropout = 0.1  # dropout probability
        self.transformer_model = network.TransformerModel(emsize, nhead, d_hid, nlayers, dropout)
        if self.cuda:
            self.transformer_model = self.transformer_model.cuda()

    def generate_square_subsequent_mask(sz: int) -> Tensor:
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

    def __initialize_training_parameters(self):
        self.criterion = CrossEntropyLoss()
        self.optimizer = Adam(self.transformer_model.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 'max', patience=8, factor=0.1, verbose=True)

    def __perf_measure(self, y_actual, y_hat):
        TP = FP = TN = FN = 0
        for i in range(len(y_hat)):
            if y_actual[i] == y_hat[i] == 1:
                TP += 1
            if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
                FP += 1
            if y_actual[i] == y_hat[i] == 0:
                TN += 1
            if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
                FN += 1

        return (TP, FP, TN, FN)

    def __train_epoch(self, epoch):
        self.transformer_model.train()
        losses = []
        accuracies = []
        TN = FN = TP = FP = 0
        all_targets = np.array([])
        predictions = np.array([])
        progress = tqdm(enumerate(self.train_loader), total=len(
            self.train_loader), desc='Training', file=sys.stdout)
        for batch_idx, data in progress:
            samples, targets = data
            samples, targets = self.samples_to_cuda(samples, targets)

            features, targets = self.process_samples_through_cnn(samples, targets)

            outputs = self.transformer_model(features)

            loss = self.criterion(outputs, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())
            targets = targets.data.cpu()
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.data.cpu()
            all_targets = np.concatenate((all_targets, targets))
            predictions = np.concatenate((predictions, predicted))
            perf = self.__perf_measure(targets, predicted)
            TN += perf[2]
            FN += perf[3]
            TP += perf[0]
            FP += perf[1]

            acc = (TP + TN) / (FP + FN + TP + TN)
            precision = TP / (TP + FP) if TP + FP > 0 else 0
            recall = TP / (TP + FN) if TP + FN > 0 else 0
            f1 = 2 * (precision * recall) / (precision +
                                             recall) if precision + recall > 0 else 0
            # TODO wrong calculation for 3 classes
            precision, recall, f1, _ = metrics.precision_recall_fscore_support(all_targets, predictions,
                                                                               average="macro")
            acc = metrics.accuracy_score(all_targets, predictions)
            accuracies.append(acc)
            progress.set_description(
                'Training Loss: {:.4f} | Accuracy: {:.4f} | F1: {:.4f} | Precision: {:.4f} | Recall: {:.4f} | TP: {} | TN: {} | FP: {} | FN: {}'.format(
                    loss.item(), acc, f1, precision, recall, TP, TN, FP, FN))

        train_stats = {'loss': np.mean(losses), 'accuracy': np.mean(accuracies), 'precision': precision,
                       'recall': recall, 'f1': f1}
        print('Epoch: {}/{}, Average train loss: {:.4f}, Average train accuracy: {:.4f}'.format(
            epoch, self.epochs, train_stats['loss'], train_stats['accuracy']))
        return train_stats

    def moving_average(self, values):
        #return np.append(np.average(sliding_window_view(values, window_shape = n), axis=1), np.zeros(n-1))
        return np.concatenate((np.zeros(2) ,np.average(sliding_window_view(values, window_shape = 5), axis=1), np.zeros(2)), axis=None)

    def process_samples_through_cnn(self, samples, targets):
        features = self.cnn_model.get_embedding(samples)
        features, targets = self.__fill_features_if_needed(features, targets)
        features = features.reshape(-1, self.sequence_len, self.dims)

        return features, targets

    def samples_to_cuda(self, samples, targets):
        if self.cuda:
            samples = samples.cuda()
            targets = targets.cuda()
        return samples, targets

    def __fill_features_if_needed(self, features, targets):
        if features.numel() < self.dims * self.batch_size:
            zeros_features = torch.zeros(
                self.batch_size - features.shape[0], self.dims)
            if self.cuda:
                zeros_features = zeros_features.cuda()
            features = torch.cat((features, zeros_features))
            zeros_targets = torch.zeros(
                self.batch_size - targets.shape[0], dtype=torch.long)
            if self.cuda:
                zeros_targets = zeros_targets.cuda()
            targets = torch.cat((targets, zeros_targets))
        return features, targets

    def gpu_warm_up(self):
        dummy_input = torch.randn(512, 3, 100, 100, dtype=torch.float).cuda()
        for _ in range(100):
            _ = self.cnn_model.get_embedding(dummy_input)

    def __test_epoch(self, evaluation=False):
        self.transformer_model.eval()

        losses = []
        accuracies = []
        TN = FN = TP = FP = 0
        progress = tqdm(enumerate(self.test_loader), total=len(
            self.test_loader), desc='Testing', file=sys.stdout)
        predictions = np.array([])
        all_targets = np.array([])
        all_probabilities = np.array([])

        total_time = 0
        # self.gpu_warm_up()
        for batch_idx, data in progress:
            with torch.no_grad():
                samples, targets = data
                samples, targets = self.samples_to_cuda(samples, targets)

                starter = torch.cuda.Event(enable_timing=True)
                ender = torch.cuda.Event(enable_timing=True)

                starter.record()
                features, targets = self.process_samples_through_cnn(samples, targets)
                outputs = self.transformer_model(features)

                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender) / 1000
                total_time += curr_time

                loss = self.criterion(outputs, targets)
                losses.append(loss.item())

                targets = targets.data.cpu()

                probabilities = outputs[:,1].data.cpu()
                _, predicted = torch.max(outputs.data, 1)
                predicted = predicted.data.cpu()
                all_targets = np.concatenate((all_targets, targets))
                predictions = np.concatenate((predictions, predicted))
                all_probabilities = np.concatenate((all_probabilities, probabilities))

                perf = self.__perf_measure(targets, predicted)
                TN += perf[2]
                FN += perf[3]
                TP += perf[0]
                FP += perf[1]

                acc = (TP + TN) / (FP + FN + TP + TN)
                accuracies.append(acc)
                precision = TP / (TP + FP) if TP + FP > 0 else 0
                recall = TP / (TP + FN) if TP + FN > 0 else 0
                f1 = 2 * (precision * recall) / (precision +
                                                 recall) if precision + recall > 0 else 0
                precision, recall, f1, _ = metrics.precision_recall_fscore_support(all_targets, predictions,
                                                                                   average="macro")
                acc = metrics.accuracy_score(all_targets, predictions)
                progress.set_description(
                    'Test Loss: {:.4f} | Accuracy: {:.4f} | F1: {:.4f} | Precision: {:.4f} | Recall: {:.4f} | TP: {} | TN: {} | FP: {} | FN: {}'.format(
                        loss.item(), acc, f1, precision, recall, TP, TN, FP, FN))

        dataframe = self.test_loader.dataset.getDataframe().copy()
        predictions = predictions[:len(dataframe)]
        all_targets = all_targets[:len(dataframe)]
        all_probabilities = all_probabilities[:len(dataframe)]
        print('Total Elapsed Processing time: ', total_time)
        print('Total processed elements: ', len(predictions))
        print('Time per element', total_time / len(predictions))

        moving_avg_probabilities = self.moving_average(all_probabilities)
        moving_avg_preds = moving_avg_probabilities > 0.7
        dataframe['pred'] = moving_avg_preds
        dataframe['target'] = all_targets
        print(metrics.classification_report(all_targets, moving_avg_preds))
        print(metrics.classification_report(all_targets, predictions))
        print(self.evaluator.evaluate(dataframe))

        dataframe['pred'] = predictions
        dataframe['target'] = all_targets

        #print(metrics.classification_report(all_targets, predictions))
        confussion_matrix = metrics.confusion_matrix(all_targets, predictions).ravel()
        print(confussion_matrix)
        # if evaluation:
        dataframe.to_csv('-'.join(self.params.get('test_dataset_dirs')) + '-' + self.eval_mode + '.csv', index=False)
        return self.evaluator.evaluate(dataframe)

    def eval_performance(self):
        self.transformer_model.eval()
        print(self.transformer_model)

        total_time_cnn = 0
        total_time_lstm = 0
        self.gpu_warm_up()
        repetitions = 2000
        for _ in range(repetitions):
            with torch.no_grad():
                dummy_samples = torch.randn(self.batch_size, 3, 100, 100, dtype=torch.float).cuda()
                dummy_targets = torch.zeros(self.batch_size, 1).cuda()

                starter_cnn = torch.cuda.Event(enable_timing=True)
                ender_cnn = torch.cuda.Event(enable_timing=True)

                starter_lstm = torch.cuda.Event(enable_timing=True)
                ender_lstm = torch.cuda.Event(enable_timing=True)

                starter_cnn.record()
                features, targets = self.process_samples_through_cnn(dummy_samples, dummy_targets)

                ender_cnn.record()
                torch.cuda.synchronize()
                elapsed_time_cnn = starter_cnn.elapsed_time(ender_cnn)

                starter_lstm.record()
                outputs = self.transformer_model(features)

                ender_lstm.record()
                torch.cuda.synchronize()
                elapsed_time_lstm = starter_lstm.elapsed_time(ender_lstm)
                total_time_cnn += elapsed_time_cnn
                total_time_lstm += elapsed_time_lstm

        total_time = total_time_cnn + total_time_lstm
        print('Total processed elements: ', self.batch_size * repetitions)
        print('Total Elapsed Processing time: ', total_time)
        print('Time per element', total_time / (self.batch_size * repetitions))
        print('Total Elapsed Procesing time (CNN): ', total_time_cnn)
        print('Time per element (CNN)', total_time_cnn / (self.batch_size * repetitions))
        print('Total Elapsed Procesing time (LSTM): ', total_time_lstm)
        print('Time per element (LSTM)', total_time_lstm / (self.batch_size * repetitions))

    def eval(self):
        self.transformer_model.load_state_dict(torch.load(self.transformer_model_file))
        self.transformer_model.eval()
        if self.cuda:
            self.transformer_model = self.transformer_model.cuda()
        results = self.__test_epoch(True)
        self.print_eval_results(results)

    def print_eval_results(self, results):
        print('Eval results => F1: {:.4f} | Precision: {:.4f} | Recall: {:.4f} | TP: {} | FP: {} | FN: {}'.format(
            results['f1'], results['precision'], results['recall'], results['tp'], results['fp'], results['fn']))

    def fit(self):
        for epoch in range(1, self.epochs + 1):
            self.scheduler.step(self.current_f1)
            train_stats = self.__train_epoch(epoch)

            test_stats = self.__test_epoch()
            self.eval_training(epoch, train_stats, test_stats)

            if self.current_f1 > self.best_f1:
                self.best_f1 = self.current_f1
                self.best_epoch = epoch
                torch.save(self.transformer_model.state_dict(),
                           self.base_file_name + '.pt')
            torch.save(self.transformer_model.state_dict(), 'last_model.pt')
            print('Current best model -> Epoch {} F1: {}'.format(self.best_epoch, self.best_f1))

        self.log_file.close()

    def eval_training(self, epoch, train_stats, eval_stats):
        self.current_f1 = eval_stats['f1']
        print('Epoch: {}/{}, F1: {:.4f} | Precision: {:.4f} | Recall: {:.4f} | TP: {} | FP: {} | FN: {}'.format(
            epoch, self.epochs, eval_stats['f1'], eval_stats['precision'], eval_stats['recall'], eval_stats['tp'],
            eval_stats['fp'], eval_stats['fn']))
        # self.log_file.write('{},{},{},{},{},{},{},{},{},{},{},{},{}\n'
        #                     .format(epoch,
        #                             train_stats['loss'],
        #                             train_stats['accuracy'],
        #                             train_stats['precision'],
        #                             train_stats['recall'],
        #                             train_stats['f1'],
        #                             eval_stats['f1'],
        #                             eval_stats['precision'],
        #                             eval_stats['recall'],
        #                             eval_stats['tp'],
        #                             eval_stats['fp'],
        #                             eval_stats['db'],
        #                             eval_stats['fn']
        #                             ))


class BlinkDetectionTransformerModel(TransformerModel):

    def __init__(self, params, cuda):
        self.num_classes = 2
        super().__init__(params, evaluator.BlinkDetectionEvaluator(), cuda)

    def initialize_train_loader(self):
        self.train_set = dataloader.BlinkDetectionLSTMDataset(
            self.train_dataset_dirs, self.TRAIN_TRANSFORM, videos=self.train_videos)
        self.train_loader = DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=False, num_workers=8)

    def initialize_evaluation_loader(self):
        self.test_set = dataloader.BlinkDetectionLSTMDataset(
            self.test_dataset_dirs, self.TEST_TRANSFORM, videos=self.test_videos)
        self.test_loader = DataLoader(
            self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=8)


class EyeStateDetectionSingleInputTransformerModel(TransformerModel):

    def __init__(self, params, cuda):
        self.num_classes = 2
        super().__init__(params, evaluator.EyeStateDetectionEvaluator(), cuda)

    def initialize_train_loader(self):
        self.train_set = dataloader.EyeStateDetectionSingleInputLSTMDataset(
            self.train_dataset_dirs, self.TRAIN_TRANSFORM, videos=self.train_videos)
        self.train_loader = DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=False, num_workers=8)

    def initialize_evaluation_loader(self):
        self.test_set = dataloader.EyeStateDetectionSingleInputLSTMDataset(
            self.test_dataset_dirs, self.TEST_TRANSFORM, videos=self.test_videos)
        self.test_loader = DataLoader(
            self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=8)


class EyeStateDetectionTransformerModel(TransformerModel):

    def __init__(self, params, cuda):
        self.num_classes = 2
        super().__init__(params, evaluator.EyeStateDetectionEvaluator(), cuda)

    def initialize_train_loader(self):
        self.train_set = dataloader.EyeStateDetectionLSTMDataset(
            self.train_dataset_dirs, self.TRAIN_TRANSFORM, videos=self.train_videos)
        self.train_loader = DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=False, num_workers=8)

    def initialize_evaluation_loader(self):
        self.test_set = dataloader.EyeStateDetectionLSTMDataset(
            self.test_dataset_dirs, self.TEST_TRANSFORM, videos=self.test_videos)
        self.test_loader = DataLoader(
            self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=8)

    def initialize_transformer_model(self):
        self.transformer_model = network.BiRNN(
            2 * self.dims, self.lstm_hidden_units, self.lstm_layers, self.num_classes)
        if self.cuda:
            self.transformer_model = self.transformer_model.cuda()

    def process_samples_through_cnn(self, samples, targets):
        left_eyes, right_eyes = samples
        # targets_left and targets_right are equal
        left_eye_features, targets_left = super().process_samples_through_cnn(left_eyes, targets)
        right_eye_features, targets_right = super().process_samples_through_cnn(right_eyes, targets)

        concatenation = torch.cat((left_eye_features, right_eye_features), 2)
        return concatenation, targets_left

    def samples_to_cuda(self, samples, targets):
        left_eyes, right_eyes = samples
        if self.cuda:
            targets = targets.cuda()
            left_eyes = left_eyes.cuda()
            right_eyes = right_eyes.cuda()

        return (left_eyes, right_eyes), targets

    def eval_performance(self):
        self.transformer_model.eval()
        print(self.transformer_model)

        total_time_cnn = 0
        total_time_lstm = 0
        self.gpu_warm_up()
        repetitions = 2000
        for _ in range(repetitions):
            with torch.no_grad():
                dummy_samples_left = torch.randn(self.batch_size, 3, 100, 100, dtype=torch.float).cuda()
                dummy_samples_right = torch.randn(self.batch_size, 3, 100, 100, dtype=torch.float).cuda()
                dummy_samples = (dummy_samples_left, dummy_samples_right)
                dummy_targets = torch.zeros(self.batch_size, 1).cuda()

                starter_cnn = torch.cuda.Event(enable_timing=True)
                ender_cnn = torch.cuda.Event(enable_timing=True)

                starter_lstm = torch.cuda.Event(enable_timing=True)
                ender_lstm = torch.cuda.Event(enable_timing=True)

                starter_cnn.record()
                features, targets = self.process_samples_through_cnn(dummy_samples, dummy_targets)

                ender_cnn.record()
                torch.cuda.synchronize()
                elapsed_time_cnn = starter_cnn.elapsed_time(ender_cnn)

                starter_lstm.record()
                outputs = self.transformer_model(features)

                ender_lstm.record()
                torch.cuda.synchronize()
                elapsed_time_lstm = starter_lstm.elapsed_time(ender_lstm)
                total_time_cnn += elapsed_time_cnn
                total_time_lstm += elapsed_time_lstm

        total_time = total_time_cnn + total_time_lstm
        print('Total processed elements: ', self.batch_size * repetitions)
        print('Total Elapsed Processing time: ', total_time)
        print('Time per element', total_time / (self.batch_size * repetitions))
        print('Total Elapsed Procesing time (CNN): ', total_time_cnn)
        print('Time per element (CNN)', total_time_cnn / (self.batch_size * repetitions))
        print('Total Elapsed Procesing time (LSTM): ', total_time_lstm)
        print('Time per element (LSTM)', total_time_lstm / (self.batch_size * repetitions))


class BlinkCompletenessDetectionTransformerModel(TransformerModel):

    def __init__(self, params, cuda):
        self.num_classes = 3
        super().__init__(params, evaluator.BlinkCompletenessDetectionEvaluator(), cuda)
        self.LOG_FILE_HEADER = 'EPOCH,TRAIN_LOSS,TRAIN_ACCURACY,TRAIN_PRECISION,TRAIN_RECALL,TRAIN_F1,PARTIAL_F1,PARTIAL_PRECISION,PARTIAL_RECALL,PARTIAL_TP,PARTIAL_FP,PARTIAL_TN,PARTIAL_FN,COMPLETE_F1,COMPLETE_PRECISION,COMPLETE_RECALL,COMPLETE_TP,COMPLETE_FP,COMPLETE_TN,COMPLETE_FN'

    def evaluate_results(self, dataframe):
        return evaluatePartialBlinks(dataframe)

    def initialize_train_loader(self):
        self.train_set = dataloader.BlinkCompletenessDetectionLSTMDataset(
            self.train_dataset_dirs, self.TRAIN_TRANSFORM, videos=self.train_videos)
        self.train_loader = DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=False, num_workers=8)

    def initialize_evaluation_loader(self):
        self.test_set = dataloader.BlinkCompletenessDetectionLSTMDataset(
            self.test_dataset_dirs, self.TEST_TRANSFORM, videos=self.test_videos)
        self.test_loader = DataLoader(
            self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=8)

    def eval_training(self, epoch, train_stats, eval_stats):
        eval_stats_partial, eval_stats_complete = eval_stats
        self.current_f1 = eval_stats_partial['f1'] + eval_stats_complete['f1']
        print(
            'Epoch: {}/{}, Partial => F1:  {:.4f} | Precision: {:.4f} | Recall: {:.4f} | TP: {} | FP: {} | FN: {}'.format(
                epoch, self.epochs, eval_stats_partial['f1'], eval_stats_partial['precision'],
                eval_stats_partial['recall'], eval_stats_partial['tp'], eval_stats_partial['fp'],
                eval_stats_partial['fn']))
        print(
            'Epoch: {}/{}, Complete => F1: {:.4f} | Precision: {:.4f} | Recall: {:.4f} | TP: {} | FP: {} | FN: {}'.format(
                epoch, self.epochs, eval_stats_complete['f1'], eval_stats_complete['precision'],
                eval_stats_complete['recall'], eval_stats_complete['tp'], eval_stats_complete['fp'],
                eval_stats_complete['fn']))
        self.log_file.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'
                            .format(epoch,
                                    train_stats['loss'],
                                    train_stats['accuracy'],
                                    train_stats['precision'],
                                    train_stats['recall'],
                                    train_stats['f1'],
                                    eval_stats_partial['f1'],
                                    eval_stats_partial['precision'],
                                    eval_stats_partial['recall'],
                                    eval_stats_partial['tp'],
                                    eval_stats_partial['fp'],
                                    eval_stats_partial['db'],
                                    eval_stats_partial['fn'],
                                    eval_stats_complete['f1'],
                                    eval_stats_complete['precision'],
                                    eval_stats_complete['recall'],
                                    eval_stats_complete['tp'],
                                    eval_stats_complete['fp'],
                                    eval_stats_complete['db'],
                                    eval_stats_complete['fn']
                                    ))

    def print_eval_results(self, results):
        results_partial, results_complete = results
        print(
            'Eval results partial => F1: {:.4f} | Precision: {:.4f} | Recall: {:.4f} | TP: {} | FP: {} | FN: {}'.format(
                results_partial['f1'], results_partial['precision'], results_partial['recall'], results_partial['tp'],
                results_partial['fp'], results_partial['fn']))
        print(
            'Eval results complete => F1: {:.4f} | Precision: {:.4f} | Recall: {:.4f} | TP: {} | FP: {} | FN: {}'.format(
                results_complete['f1'], results_complete['precision'], results_complete['recall'],
                results_complete['tp'], results_complete['fp'], results_complete['fn']))


def create_transformer_model(params, cuda):
    eval_mode = params.get('eval_mode')
    if 'BLINK_DETECTION_MODE' == eval_mode:
        return BlinkDetectionTransformerModel(params, cuda)
    elif 'BLINK_COMPLETENESS_MODE' == eval_mode:
        return BlinkCompletenessDetectionTransformerModel(params, cuda)
    elif 'EYE_STATE_DETECTION_MODE' == eval_mode:
        return EyeStateDetectionTransformerModel(params, cuda)
    elif 'EYE_STATE_DETECTION_SINGLE_INPUT_MODE' == eval_mode:
        return EyeStateDetectionSingleInputTransformerModel(params, cuda)
    else:
        sys.exit('Unknown eval_mode=' + eval_mode)
