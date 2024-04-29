import sys
import numpy as np
from evaluator import EyeStateDetectionEvaluator
import torch
import dataloader
import network
from functional import seq
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torch.nn import BCELoss  
from torch.optim import Adam
from PIL import Image
from augmentator import ImgAugTransform
from tqdm import tqdm
from abc import ABC, abstractmethod
from torchinfo import summary


class FFModel(ABC):


    TRAIN_TRANSFORM = transforms.Compose([
        transforms.Resize((100, 100), interpolation=Image.BICUBIC),
        ImgAugTransform(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    TEST_TRANSFORM = transforms.Compose([
        transforms.Resize((100, 100), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    def __init__(self, params, cuda):
        self.params = params
        self.cuda = cuda

        self.batch_size = params.get('batch_size')
        self.dims = params.get('dims')
        self.cnn_model_file = params.get('cnn_model_file')
        self.ff_model_file = params.get('ff_model_file')
        self.epochs = params.get('epochs')
        self.eval_mode = params.get('eval_mode')
        self.lr = params.get('lr')

        if 'train_dataset_dirs' in params:
            self.train_dataset_dirs = seq(params.get('train_dataset_dirs'))\
                .map(lambda x: "{}/{}".format(params.get('datasets_base_path'), x))
        if 'test_dataset_dirs' in params:
            self.test_dataset_dirs = seq(params.get('test_dataset_dirs'))\
                .map(lambda x: "{}/{}".format(params.get('datasets_base_path'), x))
        
        self.train_videos = None if not 'train_videos' in params else params.get('train_videos')
        self.test_videos = None if not 'test_videos' in params else params.get('test_videos')

        self.best_f1 = -1
        self.best_epoch = -1
        self.current_f1 = -1

        self.__initialize_cnn_model()
        self.initialize_ff_model()
        self.__initialize_log_file()    
        self.initialize_train_loader()
        self.initialize_evaluation_loader()
        self.__initialize_training_parameters()

    def __initialize_log_file(self):
        self.LOG_FILE_HEADER = 'EPOCH,TRAIN_LOSS,TRAIN_ACCURACY,TRAIN_PRECISION,TRAIN_RECALL,TRAIN_F1,EVAL_F1,EVAL_PRECISION,EVAL_RECALL,EVAL_TP,EVAL_FP,EVAL_TN,EVAL_FN'
        separator = '-'
        self.base_file_name = "FF_train[{}]_test[{}]_batch_[{}]_dims[{}]_lr[{}]_epochs[{}]"\
            .format(separator.join(self.params.get('train_dataset_dirs')),
                    separator.join(self.params.get('test_dataset_dirs')),
                    self.batch_size,
                    self.dims,
                    self.lr,
                    self.epochs)

        self.log_file = open(self.base_file_name + '.log', 'w')

        self.log_file.write(self.base_file_name + '\n\n')
        self.log_file.write(self.LOG_FILE_HEADER)


    def initialize_train_loader(self):
        self.train_set = dataloader.EyeStateDetectionLSTMDataset(
            self.train_dataset_dirs, self.TRAIN_TRANSFORM, videos = self.train_videos)
        self.train_loader = DataLoader(
            self.train_set, batch_size=self.batch_size, num_workers=8)

    def initialize_evaluation_loader(self):
        self.test_set = dataloader.EyeStateDetectionLSTMDataset(
            self.test_dataset_dirs, self.TEST_TRANSFORM, videos = self.test_videos)
        self.test_loader = DataLoader(
            self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=8)

    def __initialize_cnn_model(self):
        self.cnn_model = network.SiameseNetV2(self.dims)
        self.cnn_model.load_state_dict(torch.load(self.cnn_model_file))
        self.cnn_model.eval()
        if self.cuda:
            self.cnn_model = self.cnn_model.cuda()

    def initialize_ff_model(self):
        self.ff_model = network.FFNet(self.dims)
        print(summary(self.ff_model, input_size=(32,512)))
        if self.cuda:
            self.ff_model = self.ff_model.cuda()
            print(summary(self.ff_model, input_size=(32,512)))

    def __initialize_training_parameters(self):
        self.criterion = BCELoss()
        self.optimizer = Adam(self.ff_model.parameters(), lr=self.params.get('lr'))

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
        self.ff_model.train()
        losses = []
        accuracies = []
        TN = FN = TP = FP = 0
        progress = tqdm(enumerate(self.train_loader), total=len(
            self.train_loader), desc='Training', file=sys.stdout)
        for batch_idx, data in progress:
            samples, targets = data
            samples, targets = self.samples_to_cuda(samples, targets)

            features, targets  = self.process_samples_through_cnn(samples, targets)

            outputs = self.ff_model(features)

            loss = self.criterion(outputs, targets.float())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())
            targets = targets.data.cpu()
            predicted =outputs.data.cpu()
            predicted = np.where(predicted > 0.5, 1 ,0)
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
            accuracies.append(acc)
            progress.set_description('Training Loss: {:.4f} | Accuracy: {:.4f} | F1: {:.4f} | Precision: {:.4f} | Recall: {:.4f} | TP: {} | TN: {} | FP: {} | FN: {}'.format(
                loss.item(), acc, f1, precision, recall, TP, TN, FP, FN))

        train_stats = {'loss': np.mean(losses), 'accuracy': np.mean(accuracies), 'precision':precision, 'recall':recall, 'f1': f1}
        print('Epoch: {}/{}, Average train loss: {:.4f}, Average train accuracy: {:.4f}'.format(
            epoch, self.epochs, train_stats['loss'], train_stats['accuracy']))
        return train_stats


    def process_samples_through_cnn(self, samples, targets):
        left_eyes, right_eyes = samples
        
        left_eye_features = self.cnn_model.get_embedding(left_eyes)
        left_eye_features, left_targets = self.fill_features_if_needed(left_eye_features, targets)
        right_eye_features = self.cnn_model.get_embedding(right_eyes)
        right_eye_features, right_targets = self.fill_features_if_needed(right_eye_features, targets)

        concatenation = torch.cat((left_eye_features, right_eye_features), 1)
        return concatenation, right_targets

    def samples_to_cuda(self, samples, targets):
        left_eyes, right_eyes = samples
        if self.cuda:
            targets = targets.cuda()
            left_eyes = left_eyes.cuda()
            right_eyes = right_eyes.cuda()
        return (left_eyes, right_eyes), targets
        



    def fill_features_if_needed(self, features, targets):
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


    def __test_epoch(self, evaluation=False):
        self.ff_model.eval()

        losses = []
        accuracies = []
        TN = FN = TP = FP = 0
        progress = tqdm(enumerate(self.test_loader), total=len(
            self.test_loader), desc='Testing', file=sys.stdout)
        predictions = np.array([])
        all_targets = np.array([])


        for batch_idx, data in progress:
            with torch.no_grad():
                samples, targets = data
                samples, targets = self.samples_to_cuda(samples, targets)

                features, targets  = self.process_samples_through_cnn(samples, targets)

                outputs = self.ff_model(features)

                loss = self.criterion(outputs, targets.float())
                losses.append(loss.item())


                losses.append(loss.item())
                targets = targets.data.cpu()
                predicted =outputs.data.cpu()
                predicted = np.where(predicted > 0.5, 1 ,0)
                perf = self.__perf_measure(targets, predicted)
                all_targets = np.concatenate((all_targets, targets))
                predictions = np.concatenate((predictions, predicted))

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
                progress.set_description('Test Loss: {:.4f} | Accuracy: {:.4f} | F1: {:.4f} | Precision: {:.4f} | Recall: {:.4f} | TP: {} | TN: {} | FP: {} | FN: {}'.format(
                    loss.item(), acc, f1, precision, recall, TP, TN, FP, FN))

        dataframe = self.test_loader.dataset.getDataframe().copy()
        predictions = predictions[:len(dataframe)]
        all_targets = all_targets[:len(dataframe)]
        dataframe['pred'] = predictions
        dataframe['target'] = all_targets
        if evaluation:
            dataframe.to_csv('-'.join(self.params.get('test_dataset_dirs')) + '-' + self.eval_mode + '.csv', index=False)
        return EyeStateDetectionEvaluator().evaluate(dataframe)

    def gpu_warm_up(self):
        dummy_input = torch.randn(self.batch_size, 3 ,100, 100, dtype=torch.float).cuda()
        for _ in range(100):
            _ = self.cnn_model.get_embedding(dummy_input)
    
    def eval_performance(self):
        self.ff_model.eval()

        total_time_cnn = 0
        total_time_ff = 0
        self.gpu_warm_up()
        repetitions = 2000
        for _ in range(repetitions):
            with torch.no_grad():
                dummy_samples_left = torch.randn(self.batch_size, 3 ,100, 100, dtype=torch.float).cuda()
                dummy_samples_right = torch.randn(self.batch_size, 3 ,100, 100, dtype=torch.float).cuda()
                dummy_samples = (dummy_samples_left, dummy_samples_right)
                dummy_targets = torch.zeros(self.batch_size, 1).cuda()

                starter_cnn = torch.cuda.Event(enable_timing=True)
                ender_cnn = torch.cuda.Event(enable_timing=True)
                
                starter_ff = torch.cuda.Event(enable_timing=True)
                ender_ff = torch.cuda.Event(enable_timing=True)
                
                starter_cnn.record()
                features, targets  = self.process_samples_through_cnn(dummy_samples, dummy_targets)

                ender_cnn.record()
                torch.cuda.synchronize()
                elapsed_time_cnn = starter_cnn.elapsed_time(ender_cnn)

                starter_ff.record()
                outputs = self.ff_model(features)

                ender_ff.record()
                torch.cuda.synchronize()
                elapsed_time_ff = starter_ff.elapsed_time(ender_ff)
                total_time_cnn += elapsed_time_cnn
                total_time_ff += elapsed_time_ff

        total_time = total_time_cnn + total_time_ff
        print('Total processed elements: ', self.batch_size * repetitions)
        print('Total Elapsed Processing time: ', total_time)
        print('Time per element', total_time / (self.batch_size * repetitions))
        print('Total Elapsed Procesing time (CNN): ', total_time_cnn)
        print('Time per element (CNN)', total_time_cnn / (self.batch_size * repetitions))
        print('Total Elapsed Procesing time (FF): ', total_time_ff)
        print('Time per element (FF)', total_time_ff / (self.batch_size * repetitions))

    def eval(self):
        self.ff_model.load_state_dict(torch.load(self.ff_model_file))
        self.ff_model.eval()
        if self.cuda:
            self.ff_model = self.ff_model.cuda()
        results = self.__test_epoch(True)
        self.print_eval_results(results)
    
    def print_eval_results(self, results):
        print('Eval results => F1: {:.4f} | Precision: {:.4f} | Recall: {:.4f} | TP: {} | FP: {} | FN: {}'.format(
            results['f1'], results['precision'], results['recall'], results['tp'], results['fp'], results['fn']))


    def fit(self):
        for epoch in range(1, self.epochs + 1):
            train_stats = self.__train_epoch(epoch)

            test_stats = self.__test_epoch()
            self.eval_training(epoch, train_stats, test_stats)

            if self.current_f1 > self.best_f1:
                self.best_f1 = self.current_f1
                self.best_epoch = epoch
                torch.save(self.ff_model.state_dict(),
                           self.base_file_name + '.pt')
            torch.save(self.ff_model.state_dict(), 'last_model.pt')
            print('Current best model -> Epoch {} F1: {}'.format(self.best_epoch, self.best_f1))

        self.log_file.close()


    def eval_training(self, epoch, train_stats, eval_stats):
        self.current_f1 = eval_stats['f1']
        print('Epoch: {}/{}, F1: {:.4f} | Precision: {:.4f} | Recall: {:.4f} | TP: {} | FP: {} | FN: {}'.format(
            epoch, self.epochs, eval_stats['f1'], eval_stats['precision'], eval_stats['recall'], eval_stats['tp'], eval_stats['fp'], eval_stats['fn']))
        self.log_file.write('{},{},{},{},{},{},{},{},{},{},{},{},{}\n'
                            .format(epoch,
                                    train_stats['loss'],
                                    train_stats['accuracy'],
                                    train_stats['precision'],
                                    train_stats['recall'],
                                    train_stats['f1'],
                                    eval_stats['f1'],
                                    eval_stats['precision'],
                                    eval_stats['recall'],
                                    eval_stats['tp'],
                                    eval_stats['fp'],
                                    eval_stats['db'],
                                    eval_stats['fn']
                                    ))


class EyeStateDetectionSingleInputFFModel(FFModel):

    def initialize_train_loader(self):
        self.train_set = dataloader.EyeStateDetectionSingleInputLSTMDataset(
                self.train_dataset_dirs, self.TRAIN_TRANSFORM, videos = self.train_videos)
        self.train_loader = DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=False, num_workers=8)

    def initialize_evaluation_loader(self):
        self.test_set = dataloader.EyeStateDetectionSingleInputLSTMDataset(
                self.test_dataset_dirs, self.TEST_TRANSFORM, videos = self.test_videos)
        self.test_loader = DataLoader(
            self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=8)
    

    def process_samples_through_cnn(self, samples, targets):
        features = self.cnn_model.get_embedding(samples)
        features, targets = self.fill_features_if_needed(features, targets)

        return features, targets

    def samples_to_cuda(self, samples, targets):
        if self.cuda:
            samples = samples.cuda()
            targets = targets.cuda()
        return samples, targets

    def initialize_ff_model(self):
        self.ff_model = network.FFNet(self.dims, num_inputs=1)
        if self.cuda:
            self.ff_model = self.ff_model.cuda()


def create_ff_model(params, cuda):
    eval_mode = params.get('eval_mode')
    if 'EYE_STATE_DETECTION_MODE'== eval_mode:
        return FFModel(params, cuda)
    elif 'EYE_STATE_DETECTION_SINGLE_INPUT_MODE' == eval_mode:
        return EyeStateDetectionSingleInputFFModel(params, cuda)
    else:
        sys.exit('Unknown eval_mode=' + eval_mode)
