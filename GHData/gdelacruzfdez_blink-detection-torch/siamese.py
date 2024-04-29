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
from torch.nn import BCELoss  
from sklearn.neighbors import NearestCentroid
from tqdm import tqdm
import sklearn.metrics
from sklearn.model_selection import GridSearchCV
from skorch.callbacks import PrintLog, ProgressBar
from skorch.helper import SliceDataset

class SiameseModel:

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
        self.train_dataset_dirs = seq(params.get('train_dataset_dirs'))\
            .map(lambda x: "{}/{}".format(self.params.get('datasets_base_path'), x))
        self.test_dataset_dirs = seq(params.get('test_dataset_dirs'))\
            .map(lambda x: "{}/{}".format(self.params.get('datasets_base_path'), x))
        self.best_f1 = -1
        #self.best_f1 = 1000
        self.best_epoch = -1
        self.current_f1 = -1
        #self.current_f1 = 1000

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
        self.train_set = dataloader.SiameseDataset(self.train_dataset_dirs, self.TRAIN_TRANSFORM, videos = self.train_videos)
        self.train_batch_sampler = dataloader.BalancedBatchSampler(self.train_set.targets, n_classes=len(self.train_set.classes), n_samples=self.params.get('batch_size'))
        self.train_loader = DataLoader(self.train_set, batch_sampler = self.train_batch_sampler, num_workers=8)

    def __initialize_evaluation_loader(self):
        self.test_set = dataloader.SiameseDataset(self.test_dataset_dirs, self.TEST_TRANSFORM, videos = self.test_videos)
        self.test_batch_sampler = dataloader.BalancedBatchSampler(self.test_set.targets, n_classes=len(self.test_set.classes), n_samples=self.params.get('batch_size'))
        self.test_loader = DataLoader(self.test_set, batch_sampler = self.test_batch_sampler, num_workers=8)

        #self.eval_train_set = dataloader.EyeStateDetectionSingleInputLSTMDataset(self.train_dataset_dirs, self.TEST_TRANSFORM, videos=self.train_videos)
        #self.eval_train_set = dataloader.BlinkDetectionLSTMDataset(self.train_dataset_dirs, self.TEST_TRANSFORM, videos=self.train_videos)
        self.eval_train_set = dataloader.BlinkCompletenessDetectionLSTMDataset(self.train_dataset_dirs, self.TEST_TRANSFORM, videos=self.train_videos)
        self.eval_train_loader = DataLoader(self.eval_train_set, batch_size=self.params.get('batch_size'), shuffle=False, num_workers=4)
        #self.eval_test_set = dataloader.EyeStateDetectionSingleInputLSTMDataset(self.test_dataset_dirs, self.TEST_TRANSFORM, videos=self.test_videos)
        #self.eval_test_set = dataloader.BlinkDetectionLSTMDataset(self.test_dataset_dirs, self.TEST_TRANSFORM, videos=self.test_videos)
        self.eval_test_set = dataloader.BlinkCompletenessDetectionLSTMDataset(self.test_dataset_dirs, self.TEST_TRANSFORM, videos=self.test_videos)
        self.eval_test_loader = DataLoader(self.eval_test_set, batch_size=self.params.get('batch_size'), shuffle=False, num_workers=4)

    def __initialize_model(self):
        self.model = network.SiameseNetV2(self.params.get('dims'))
        if self.cuda:
            self.model = self.model.cuda()

    def __initialize_training_parameters(self):
        self.criterion = BCELoss()
        self.optimizer = Adam(self.model.parameters(), lr=self.params.get('lr'))
        self.scheduler = StepLR(self.optimizer, 8, gamma=0.1, last_epoch=-1)

    def __train_epoch(self):
        self.model.train()
        losses = []
        progress = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc='Training', file=sys.stdout)
        for batch_idx, data in progress:
            samples, targets = data
            samples1, samples2 = samples
            if self.cuda:
                samples1 = samples1.cuda()
                samples2 = samples2.cuda()
                targets = targets.cuda()
            
            self.optimizer.zero_grad()
            outputs = self.model((samples1, samples2))
            #output1, output2 = self.model((samples1, samples2))
            loss = self.criterion(outputs, targets.float())
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())
            progress.set_description('Mean Training Loss: {:.4f}'.format(np.mean(losses)))

        return np.mean(losses)

    def __test_epoch(self):
        self.model.eval()
        losses = []
        progress = tqdm(enumerate(self.test_loader), total=len(self.test_loader), desc='Test', file=sys.stdout)
        for batch_idx, data in progress:
            with torch.no_grad():
                samples, targets = data
                samples1, samples2 = samples
                if self.cuda:
                    samples1 = samples1.cuda()
                    samples2 = samples2.cuda()
                    targets = targets.cuda()
                
                outputs = self.model((samples1, samples2))
                loss = self.criterion(outputs, targets.float())

                losses.append(loss.item())
                progress.set_description('Mean Test Loss: {:.4f}'.format(np.mean(losses)))

        return np.mean(losses)

    def __test_epoch_cluster(self):
        train_embeddings, train_targets = self.__extract_embeddings(self.eval_train_loader)
        test_embeddings, test_targets = self.__extract_embeddings(self.eval_test_loader)

        nc = NearestCentroid()
        nc.fit(train_embeddings, train_targets)
        predictions = nc.predict(test_embeddings)
        classification_report = sklearn.metrics.classification_report(test_targets, predictions, target_names=['Open','Partial','Closed'])
        #classification_report = sklearn.metrics.classification_report(test_targets, predictions, target_names=['Open','Closed'])
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
            self.scheduler.step()
            train_loss = self.__train_epoch()
            print('Epoch: {}/{}, Average train loss: {:.4f}'.format(epoch, epochs, train_loss))

            classification_report, classification_metrics , confussion_matrix = self.__test_epoch_cluster()
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
            #self.current_f1 = test_loss
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
