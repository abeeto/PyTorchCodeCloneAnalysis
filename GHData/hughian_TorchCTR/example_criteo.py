import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# from CTR.core.utils import myDataSet
from CTR.core.features import FeatureMetas
from tqdm import tqdm

from CTR.models.lr import LR
from CTR.models.fm import FM
from CTR.models.dcn import DCN
from CTR.models.pnn import PNN
from CTR.models.wd import WideDeep
from CTR.models.nfm import NFM
from CTR.models.afm import AFM
from CTR.models.deepfm import DeepFM
from CTR.models.xdeepfm import xDeepFM


class myDataSet(torch.utils.data.Dataset):
    def __init__(self, x_df: pd.DataFrame, y_df: pd.DataFrame, feature_metas: FeatureMetas):
        super(myDataSet, self).__init__()
        self.length = len(x_df)
        self.x_dense = torch.from_numpy(x_df[feature_metas.dense_names].values.astype('float32'))
        self.x_sparse = torch.from_numpy(x_df[feature_metas.sparse_names].values).long()
        self.x_varlen = torch.from_numpy(x_df[feature_metas.varlen_names].values).long()
        self.y = torch.from_numpy(y_df.values.astype('float32'))

    def __getitem__(self, index):
        return self.x_sparse[index], self.x_varlen[index], self.x_dense[index], self.y[index]

    def __len__(self):
        return self.length


def load_data():
    print('load data')
    df = pd.read_csv('./criteo_sample.csv')

    sparse_feature = ['C' + str(i) for i in range(1, 26 + 1)]
    dense_feature = ['I' + str(i) for i in range(1, 13 + 1)]
    cols = sparse_feature + dense_feature
    target = ['label']
    df[sparse_feature] = df[sparse_feature].fillna('-1')
    df[dense_feature] = df[dense_feature].fillna(0)
    df = df.sample(frac=1).reset_index(drop=True)  # shuffle
    metas = FeatureMetas()
    lbe = LabelEncoder()

    for i, name in enumerate(sparse_feature):
        df[name] = lbe.fit_transform(df[name])
        metas.add_sparse_feature(name, i, len(lbe.classes_), embedding_dim=8)

    mms = MinMaxScaler()
    df[dense_feature] = mms.fit_transform(df[dense_feature])
    for i, name in enumerate(dense_feature):
        metas.add_dense_feature(name, i + len(sparse_feature), 1, embedding_dim=8)

    train, test = train_test_split(df, test_size=0.2)
    train_set = myDataSet(train[cols], train[target], metas)
    test_set = myDataSet(test[cols], test[target], metas)
    print('data loaded')
    return train_set, test_set, metas


# TODO L1, L2 reg
def run(train, val, feature_metas, epochs=10, batch_size=256):
    # train_dataset = TensorDataset(torch.from_numpy(x_train.values), torch.from_numpy(y_train.values))
    train_loader = DataLoader(dataset=train, shuffle=True, batch_size=batch_size, num_workers=4)
    # build model
    # model = DeepFM(feature_metas)
    # model = xDeepFM(feature_metas, embedding_dim=8)
    # model = WideDeep(feature_metas)
    # model = NFM(feature_metas)
    # model = PNN(feature_metas)
    # model = AFM(feature_metas)
    model = LR(feature_metas)
    for name, param in model.named_parameters():
        print(name.ljust(60), param.shape)
    print('#' * 80)
    # optimizer
    adam = optim.Adam(model.parameters(), lr=5e-3)

    # loss fun
    criterion = nn.BCELoss(reduction='sum')
    reg1 = nn.L1Loss(reduction='sum')

    model.train()
    for epoch in range(epochs):
        print(f'Epoch: {epoch}')
        epoch_loss = 0.
        with tqdm(enumerate(train_loader), disable=False) as t:
            for i, (x_sparse, x_varlen, x_dense, y) in t:
                pred = model(x_sparse, x_varlen, x_dense).squeeze()
                adam.zero_grad()
                loss = criterion(pred, y.squeeze())
                loss += reg1(pred, y.squeeze())
                # loss += reg_loss, l1, l2
                epoch_loss += loss.item()
                loss.backward(retain_graph=True)
                adam.step()
            print('Epoch loss', epoch_loss)
            pred = model(val.x_sparse, val.x_varlen, val.x_dense)
            roc_pred = pred.detach().numpy()
            roc_true = val.y.squeeze()
            print(pred.sum(), len(pred))
            auc = roc_auc_score(roc_true, roc_pred)
            print(auc)

    return model


if __name__ == '__main__':
    train, test, feature_metas = load_data()
    run(train, test, feature_metas, batch_size=256)
