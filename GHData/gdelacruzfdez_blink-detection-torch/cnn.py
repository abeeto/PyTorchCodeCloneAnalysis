import sys
import dataloader
import network
import numpy as np
import torch
import loss
from PIL import Image
from skorch import NeuralNet
from functional import seq
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from augmentator import ImgAugTransform
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.nn import BCELoss,  CrossEntropyLoss
from sklearn.neighbors import NearestCentroid
from tqdm import tqdm
import sklearn.metrics
from sklearn.model_selection import GridSearchCV
from sklearn.utils import class_weight
from skorch.callbacks import PrintLog, ProgressBar
from skorch.helper import SliceDataset

class CNNModel:

    TRAIN_TRANSFORM =  transforms.Compose([
        transforms.Resize((100,100), interpolation=Image.BICUBIC),
        ImgAugTransform(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ])

    TEST_TRANSFORM =  transforms.Compose([
        transforms.Resize((100,100), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ])

    LOG_FILE_HEADER = 'EPOCH,TRAIN_LOSS,PRECISION,RECALL,SUPPORT,F1\n'

    def __init__(self, params, cuda):
        self.params = params
        self.cuda = cuda
        self.train_videos = None if not 'train_videos' in params else params.get('train_videos')
        self.test_videos = None if not 'test_videos' in params else params.get('test_videos')
        self.batch_size = self.params.get('batch_size')
        self.cnn_model_file = self.params.get('cnn_model_file')
        self.train_dataset_dirs = seq(params.get('train_dataset_dirs'))\
            .map(lambda x: "{}/{}".format(self.params.get('datasets_base_path'), x))
        self.test_dataset_dirs = seq(params.get('test_dataset_dirs'))\
            .map(lambda x: "{}/{}".format(self.params.get('datasets_base_path'), x))
        self.best_f1 = -1
        self.best_epoch = -1
        self.current_f1 = -1

        self.__initialize_log_file()
        self.__initialize_train_loader()
        self.__initialize_evaluation_loader()
        self.__initialize_model()
        self.__initialize_training_parameters()

    def __initialize_log_file(self):
        separator = '-'
        self.base_file_name = "CNN_train[{}]_test[{}]_batch_[{}]_dims[{}]_lr[{}]_epochs[{}]"\
            .format(separator.join(self.params.get('train_dataset_dirs')),
                    separator.join(self.params.get('test_dataset_dirs')),
                    self.params.get('batch_size'),
                    self.params.get('dims'),
                    self.params.get('lr'),
                    self.params.get('epochs'))

        self.log_file = open(self.base_file_name + '.log', 'w')
        self.log_file.write(self.base_file_name + '\n\n')
        self.log_file.write(self.LOG_FILE_HEADER)
    
    def __initialize_train_loader(self):
        #self.train_set = dataloader.BlinkCompletenessDetectionLSTMDataset(
        #        self.train_dataset_dirs, self.TRAIN_TRANSFORM, videos = self.train_videos)
        #self.train_set = dataloader.BlinkDetectionLSTMDataset(
        #        self.train_dataset_dirs, self.TRAIN_TRANSFORM, videos = self.train_videos)
        self.train_set = dataloader.EyeStateDetectionSingleInputLSTMDataset(
                self.train_dataset_dirs, self.TRAIN_TRANSFORM, videos = self.train_videos)
        self.train_loader = DataLoader(
            self.train_set, batch_size=self.batch_size, num_workers=8)

    def __initialize_evaluation_loader(self):
        #self.test_set = dataloader.BlinkCompletenessDetectionLSTMDataset(
        #        self.test_dataset_dirs, self.TEST_TRANSFORM, videos = self.test_videos)
        #self.test_set = dataloader.BlinkDetectionLSTMDataset(
        #        self.test_dataset_dirs, self.TEST_TRANSFORM, videos = self.test_videos)
        self.test_set = dataloader.EyeStateDetectionSingleInputLSTMDataset(
                self.test_dataset_dirs, self.TEST_TRANSFORM, videos = self.test_videos)
        self.test_loader = DataLoader(
            self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=8)

    def __initialize_model(self):
        self.model = network.ResNet()
        if self.cuda:
            self.model = self.model.cuda()

    def __initialize_training_parameters(self):
        dataframe = self.train_set.getDataframe()
        print('UNIQUE', np.unique(dataframe.target))
        class_weights = class_weight.compute_class_weight('balanced', np.unique(dataframe.target), dataframe.target)
        print(class_weights)
        class_weights = torch.FloatTensor(class_weights).cuda()

        self.criterion = CrossEntropyLoss(weight=class_weights)
        self.optimizer = Adam(self.model.parameters(), lr=self.params.get('lr'))
        self.scheduler = StepLR(self.optimizer, 10, gamma=0.1, last_epoch=-1)

    def __train_epoch(self):
        self.model.train()
        losses = []
        accuracies = []
        TN = FN = TP = FP = 0
        progress = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc='Training', file=sys.stdout)
        for batch_idx, data in progress:
            samples, targets = data
            if self.cuda:
                samples = samples.cuda()
                targets = targets.cuda()
            
            self.optimizer.zero_grad()
            outputs = self.model(samples)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())
            targets = targets.data.cpu()
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.data.cpu()
            perf = self.__perf_measure(targets, predicted)
            TN += perf[2]
            FN += perf[3]
            TP += perf[0]
            FP += perf[1]

            acc = (TP + TN) / (FP + FN + TP + TN) if FP + FN + TP + TN > 0 else 0
            precision = TP / (TP + FP) if TP + FP > 0 else 0
            recall = TP / (TP + FN) if TP + FN > 0 else 0
            f1 = 2 * (precision * recall) / (precision +
                                             recall) if precision + recall > 0 else 0
            accuracies.append(acc)
            progress.set_description('Training Loss: {:.4f} | Accuracy: {:.4f} | F1: {:.4f} | Precision: {:.4f} | Recall: {:.4f} | TP: {} | TN: {} | FP: {} | FN: {}'.format(
                loss.item(), acc, f1, precision, recall, TP, TN, FP, FN))

        return np.mean(losses)

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

    def __test_epoch(self):
        self.model.eval()
        losses = []
        accuracies = []
        TN = FN = TP = FP = 0
        progress = tqdm(enumerate(self.test_loader), total=len(self.test_loader), desc='Test', file=sys.stdout)
        predictions = np.array([])
        all_targets = np.array([])
        for batch_idx, data in progress:
            with torch.no_grad():
                samples, targets = data
                if self.cuda:
                    samples = samples.cuda()
                    targets = targets.cuda()
                
                outputs = self.model(samples)
                loss = self.criterion(outputs, targets)

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

                acc = (TP + TN) / (FP + FN + TP + TN) if FP + FN + TP + TN > 0 else 0
                accuracies.append(acc)
                precision = TP / (TP + FP) if TP + FP > 0 else 0
                recall = TP / (TP + FN) if TP + FN > 0 else 0
                f1 = 2 * (precision * recall) / (precision +
                                                 recall) if precision + recall > 0 else 0
                progress.set_description('Test Loss: {:.4f} | Accuracy: {:.4f} | F1: {:.4f} | Precision: {:.4f} | Recall: {:.4f} | TP: {} | TN: {} | FP: {} | FN: {}'.format(
                    loss.item(), acc, f1, precision, recall, TP, TN, FP, FN))


        #classification_report = sklearn.metrics.classification_report(all_targets, predictions, target_names=['Open','Partially Closed', 'Closed'])
        classification_report = sklearn.metrics.classification_report(all_targets, predictions, target_names=['Open', 'Closed'])
        classification_metrics = sklearn.metrics.precision_recall_fscore_support(all_targets, predictions, average='macro')
        confussion_matrix = sklearn.metrics.confusion_matrix(all_targets, predictions)

        return classification_report, classification_metrics, confussion_matrix

    def eval(self):
        self.model.load_state_dict(torch.load(self.cnn_model_file))
        self.model.eval()
        if self.cuda:
            self.model = self.model.cuda()
        results = self.__test_epoch()

    def __test_epoch_cluster(self):
        train_embeddings, train_targets = self.__extract_embeddings(self.eval_train_loader)
        test_embeddings, test_targets = self.__extract_embeddings(self.eval_test_loader)

        nc = NearestCentroid()
        nc.fit(train_embeddings, train_targets)
        predictions = nc.predict(test_embeddings)
        #classification_report = sklearn.metrics.classification_report(test_targets, predictions, target_names=['Open','Partial','Closed'])
        classification_report = sklearn.metrics.classification_report(test_targets, predictions, target_names=['Open','Closed'])
        classification_metrics = sklearn.metrics.precision_recall_fscore_support(test_targets, predictions, average='macro')
        confussion_matrix = sklearn.metrics.confusion_matrix(test_targets, predictions)

        return classification_report, classification_metrics, confussion_matrix



    def __extract_embeddings(self,loader):
        self.model.eval()

        embeddings = []
        targets = []

        with torch.no_grad():
            for sample, target in tqdm(loader, total=len(loader), desc='Testing', file=sys.stdout):
                if self.cuda:
                    sample = sample.cuda()

                output = self.model.get_embedding(sample)
                embeddings.append(output.cpu().numpy())
                targets.append(target)
        
        embeddings = np.vstack(embeddings)
        targets = np.concatenate(targets)

        return embeddings, targets


    def fit(self):
        epochs = self.params.get('epochs')
        for epoch in range(1,epochs + 1):
            #self.scheduler.step()
            train_loss = self.__train_epoch()
            print('Epoch: {}/{}, Average train loss: {:.4f}'.format(epoch, epochs, train_loss))

            #classification_report, classification_metrics , confussion_matrix = self.__test_epoch_cluster()
            classification_report, classification_metrics , confussion_matrix = self.__test_epoch()
            #test_loss = self.__test_epoch()
            print('Test Epoch: {}/{}'.format(epoch, epochs))
            #print('Train Loss: {}/ Test Loss: {}'.format(train_loss, test_loss))
            print(classification_report)
            print(classification_metrics)
            print(confussion_matrix)

            
            self.log_file.write('{},{},{},{},{},{}\n'\
                .format(epoch,
                        train_loss,
                        classification_metrics[0],
                        classification_metrics[1],
                        classification_metrics[3],
                        classification_metrics[2]))
           
            self.current_f1 = classification_metrics[2]
            if self.current_f1 > self.best_f1:
            #if self.current_f1 < self.best_f1:
                self.best_f1 = self.current_f1
                self.best_epoch = epoch
                torch.save(self.model.state_dict(), self.base_file_name + '.pt')
            print('Best model til now: Epoch {} => {}'.format(self.best_epoch, self.best_f1))
            torch.save(self.model.state_dict(), 'last_model.pt')
        self.log_file.close()
    
    def hyperparameter_tunning(self):
        net = NeuralNet(
            network.SiameseNetV2,
            max_epochs=2,
            batch_size=128,
            criterion= BCELoss,
            optimizer= Adam,
            iterator_train__num_workers=4,
            iterator_train__pin_memory=False,
            iterator_valid__num_workers=4,
            verbose=2,
            device='cuda',
            iterator_train__shuffle=True,
            callbacks=[PrintLog(), ProgressBar()])
        
        net.set_params(train_split=False)
        params = {
            'lr': [0.01, 0.001],
            'module__num_dims': [128, 256]
        }
        gs = GridSearchCV(net, params, refit=False, cv=3, scoring='f1')
        X_sl = SliceDataset(self.train_set, idx=0)
        Y_sl = SliceDataset(self.train_set, idx=1)
        gs.fit(X_sl, Y_sl)
