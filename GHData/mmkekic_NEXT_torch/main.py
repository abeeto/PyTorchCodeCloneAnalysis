import os
import yaml
from argparse     import ArgumentParser
from src.utils import train_utils as tu
from src.utils import train_utils_dann as tud
from src.utils import train_utils_mcd as tumcd
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import shutil


def is_valid_conf_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        with open(arg, 'r') as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError:
                parser.error("The file %s not valid configuration!" % arg)
def is_valid_action(parser, arg):
    if not arg in ['train', 'predict']:
        parser.error("The action %s is not allowed!" % arg)
    else:
        return arg
DataTypes = ['sparse', 'dense']
NetTypes = ['classifier', 'dann', 'MCD']

dataset = '/lustre/ific.uv.es/ml/ific020/dataset_labels.h5'
sidebands_data = '/lustre/ific.uv.es/ml/ific020/sidebands_1trck.csv'

norms_above = '/lustre/ific.uv.es/ml/ific020/norm_sidebands_above_1trck_fid.csv'
norms_below = '/lustre/ific.uv.es/ml/ific020/norm_sidebands_below_1trck_fid.csv'

def train_classifier(model, data_type, lr, n_epochs, num_iter, batch_size, folder_name, q, num_workers, augmentation, conf_file, normalize, one_track, fiducial):
    data_df_tr = pd.read_hdf(dataset, key='data_train').sample(frac=1.).reset_index(drop=True)
    data_df_ts = pd.read_hdf(dataset, key='data_valid').sample(frac=1.).reset_index(drop=True)
    data_df    = pd.concat([data_df_tr, data_df_ts], ignore_index=True)
    data_df = data_df[data_df.evt_ntrks==1].reset_index(drop=True)
    data_df = data_df[data_df.evt_r_max<190].reset_index(drop=True)
    data_df.event = data_df.event.astype(int)
    data_df.run_number = data_df.run_number.astype(int)

    selection_df = pd.read_csv(sidebands_data)
    selection_df.event = selection_df.event.astype(int)
    selection_df.run_number = selection_df.run_number.astype(int)
    train_df  = pd.read_hdf(dataset, key='MC_train').sample(frac=1.).reset_index(drop=True)
    train_df = train_df[(train_df.evt_energy<1.75) & (train_df.evt_energy>1.45)].reset_index(drop=True)
    if one_track:
        train_df = train_df[train_df.evt_ntrks==1].reset_index(drop=True)
    if fiducial:
        train_df = train_df[train_df.evt_r_max<190].reset_index(drop=True)
    valid_df  = pd.read_hdf(dataset, key='MC_valid').sample(frac=1.).reset_index(drop=True)
    valid_df = valid_df[(valid_df.evt_energy<1.75) & (valid_df.evt_energy>1.45)].reset_index(drop=True)
    if one_track:
        valid_df = valid_df[valid_df.evt_ntrks==1].reset_index(drop=True)
    if fiducial:
        valid_df = valid_df[valid_df.evt_r_max<190].reset_index(drop=True)

    valid_df.event = valid_df.event.astype(int)

    sig_ratio = train_df.label.sum()/len(train_df)
    print(f'training data stats: length {len(train_df)}, signal ration {sig_ratio}')
    weights = torch.tensor([sig_ratio, 1.-sig_ratio],device='cuda').float()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
    criterion = nn.CrossEntropyLoss(weights)
    model_path = folder_name
    os.makedirs(model_path+'/')
    os.makedirs(model_path+'/distribution_plots/')
    os.makedirs(model_path+'/logs')
    if conf_file is not None:
        shutil.copyfile(conf_file, f'{model_path}/train.conf')
    if normalize:
        norm_above = pd.read_csv(norms_above, index_col=False)[['mean', 'std', 'domain']]
        norm_below = pd.read_csv(norms_below, index_col=False)[['mean', 'std', 'domain']]
        norm = norm_above.merge(norm_below, on='domain', suffixes=['_a', '_b'])
        norm.loc[:, 'mean'] = 0.5*(norm['mean_a'] + norm['mean_b'])
        norm.loc[:, 'std' ] = 0.5*(norm['std_a' ] + norm['std_b'])
        if data_type=='dense':
            mean_MC, std_MC =  norm[norm['domain'] == 'MC_dense']['mean'].values[0], norm[norm['domain'] == 'MC_dense']['std'].values[0]
            mean_data, std_data =  norm[norm['domain'] == 'data_dense']['mean'].values[0], norm[norm['domain'] == 'data_dense']['std'].values[0]
        elif data_type=='sparse':
            mean_MC, std_MC =  norm[norm['domain'] == 'MC_sparse']['mean'].values[0], norm[norm['domain'] == 'MC_sparse']['std'].values[0]
            mean_data, std_data =  norm[norm['domain'] == 'data_sparse']['mean'].values[0], norm[norm['domain'] == 'data_sparse']['std'].values[0]
        print(mean_MC, std_MC, mean_data, std_data)
    else:
        mean_MC = None
        mean_data = None
        std_MC = None
        std_data = None
    # if data_type == 'dense':
    #     norm_series = pd.read_csv(dense_norm, header=None, index_col=False)
    #     norm_series.set_index(0, inplace=True)
    #     mean_MC, std_MC = norm_series.loc['mean'].values[0], norm_series.loc['std'].values[0]
    # else:
    #     mean, std= None, None
    tu.train(net=model, data_type=data_type, criterion=criterion, optimizer=optimizer, 
             scheduler=scheduler, batch_size=batch_size, nb_epoch=n_epochs, num_iter=num_iter, 
             train_df=train_df, valid_df=valid_df, data_df=data_df, selection_df=selection_df,
             q=q, num_workers=num_workers, model_path=folder_name, augmentation=augmentation,
             mean_MC=mean_MC, std_MC=std_MC, mean_data=mean_data, std_data=std_data
             )


def train_mcd(G, C1, C2, data_type, lr, n_epochs, num_iter, batch_size, folder_name, q, num_workers, augmentation, conf_file, normalize, one_track, fiducial, n_k):
    data_df_tr = pd.read_hdf(dataset, key='data_train').sample(frac=1.).reset_index(drop=True)
    data_df_ts = pd.read_hdf(dataset, key='data_valid').sample(frac=1.).reset_index(drop=True)
    data_df    = pd.concat([data_df_tr, data_df_ts], ignore_index=True)
    Z_corr_factor = 2.76e-4 
    data_df.loc[:,'evt_energy_z'] = data_df.evt_energy/(1. - Z_corr_factor*(data_df.evt_z_max-data_df.evt_z_min))
    data_df.loc[:,'evt_energy_z'] = 0.96 * data_df['evt_energy_z']
    #remove peak events
    data_df = data_df[(data_df.evt_energy_z<1.55) | (data_df.evt_energy_z>1.65)].reset_index(drop=True)
    if one_track:
        data_df = data_df[data_df.evt_ntrks==1].reset_index(drop=True)
    if fiducial:
        data_df = data_df[data_df.evt_r_max<190].reset_index(drop=True)
    data_df.event = data_df.event.astype(int)
    data_df.run_number = data_df.run_number.astype(int)
    print('data shape', data_df.shape)
    selection_df = pd.read_csv(sidebands_data)
    selection_df.event = selection_df.event.astype(int)
    selection_df.run_number = selection_df.run_number.astype(int)
    train_df  = pd.read_hdf(dataset, key='MC_train').sample(frac=1.).reset_index(drop=True)
    train_df = train_df[(train_df.evt_energy<1.75) & (train_df.evt_energy>1.45)].reset_index(drop=True)
    if one_track:
        train_df = train_df[train_df.evt_ntrks==1].reset_index(drop=True)
    if fiducial:
        train_df = train_df[train_df.evt_r_max<190].reset_index(drop=True)
    valid_df  = pd.read_hdf(dataset, key='MC_valid').sample(frac=1).reset_index(drop=True)
    if one_track:
        valid_df = valid_df[valid_df.evt_ntrks==1].reset_index(drop=True)
    if fiducial:
        valid_df = valid_df[valid_df.evt_r_max<190].reset_index(drop=True)

    valid_df.event = valid_df.event.astype(int)
    sig_ratio = train_df.label.sum()/len(train_df)
    print(f'training data stats: length {len(train_df)}, signal ration {sig_ratio}')
    weights = torch.tensor([sig_ratio, 1.-sig_ratio],device='cuda').float()
    criterion = nn.CrossEntropyLoss(weights)
    model_path = folder_name
    os.makedirs(model_path+'/')
    os.makedirs(model_path+'/distribution_plots/')
    os.makedirs(model_path+'/logs')
    if conf_file is not None:
        shutil.copyfile(conf_file, f'{model_path}/train.conf')
    if normalize:
        norm_above = pd.read_csv(norms_above, index_col=False)[['mean', 'std', 'domain']]
        norm_below = pd.read_csv(norms_below, index_col=False)[['mean', 'std', 'domain']]
        norm = norm_above.merge(norm_below, on='domain', suffixes=['_a', '_b'])
        norm.loc[:, 'mean'] = 0.5*(norm['mean_a'] + norm['mean_b'])
        norm.loc[:, 'std' ] = 0.5*(norm['std_a' ] + norm['std_b'])
        if data_type=='dense':
            mean_MC, std_MC =  norm[norm['domain'] == 'MC_dense']['mean'].values[0], norm[norm['domain'] == 'MC_dense']['std'].values[0]
            mean_data, std_data =  norm[norm['domain'] == 'data_dense']['mean'].values[0], norm[norm['domain'] == 'data_dense']['std'].values[0]
        elif data_type=='sparse':
            mean_MC, std_MC =  norm[norm['domain'] == 'MC_sparse']['mean'].values[0], norm[norm['domain'] == 'MC_sparse']['std'].values[0]
            mean_data, std_data =  norm[norm['domain'] == 'data_sparse']['mean'].values[0], norm[norm['domain'] == 'data_sparse']['std'].values[0]
        print(mean_MC, std_MC, mean_data, std_data)
    else:
        mean_MC = None
        mean_data = None
        std_MC = None
        std_data = None

    tumcd.train(G=G, C1=C1, C2=C2, data_type=data_type, criterion=criterion, 
                batch_size=batch_size, nb_epoch=n_epochs, lr=lr,num_iter=num_iter, 
                train_df=train_df, valid_df=valid_df, data_df=data_df, selection_df=selection_df,
                q=q, num_workers=num_workers, model_path=folder_name, augmentation=augmentation,
                mean_MC=mean_MC, std_MC=std_MC, mean_data=mean_data, std_data=std_data, n_k=n_k
            )





def train_dann(model, data_type, lr, n_epochs, batch_size, folder_name, q, num_workers):
    train_df  = pd.read_hdf(dataset, key='MC_train').sample(frac=1.).reset_index(drop=True)
    valid_df  = pd.read_hdf(dataset, key='MC_valid').sample(frac=1.).reset_index(drop=True)
    data_df  = pd.read_hdf(dataset, key='data_train').sample(frac=1.).reset_index(drop=True)
    valid_data_df  = pd.read_hdf(dataset, key='data_valid').sample(frac=1.).reset_index(drop=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
    criterion = nn.CrossEntropyLoss()
    model_path = folder_name
    os.makedirs(model_path)
    os.makedirs(model_path+'/logs')
    tud.train(net=model, data_type=data_type, criterion=criterion, optimizer=optimizer, 
             scheduler=scheduler, batch_size=batch_size, nb_epoch=n_epochs, 
             train_df=train_df, valid_df=valid_df, data_df=data_df, valid_data_df=valid_data_df,
              q=q, num_workers=num_workers, model_path=folder_name, augmentation=True)

# MC
def pred_MC(model, net_type, datatype, q, batch_size, num_workers, folder, format_name, normalize, sidebands_features=False):
    df_test  = pd.read_hdf(dataset, key='MC_valid')
    df_test_prime = df_test.drop(['label'], axis=1)
    if normalize:
        norm_above = pd.read_csv(norms_above, index_col=False)[['mean', 'std', 'domain']]
        norm_below = pd.read_csv(norms_below, index_col=False)[['mean', 'std', 'domain']]
        norm = norm_above.merge(norm_below, on='domain', suffixes=['_a', '_b'])
        norm.loc[:, 'mean'] = 0.5*(norm['mean_a'] + norm['mean_b'])
        norm.loc[:, 'std' ] = 0.5*(norm['std_a' ] + norm['std_b'])
        if data_type=='dense':
            mean_MC, std_MC =  norm[norm['domain'] == 'MC_dense']['mean'].values[0], norm[norm['domain'] == 'MC_dense']['std'].values[0]
            mean_data, std_data =  norm[norm['domain'] == 'data_dense']['mean'].values[0], norm[norm['domain'] == 'data_dense']['std'].values[0]
        elif data_type=='sparse':
            mean_MC, std_MC =  norm[norm['domain'] == 'MC_sparse']['mean'].values[0], norm[norm['domain'] == 'MC_sparse']['std'].values[0]
            mean_data, std_data =  norm[norm['domain'] == 'data_sparse']['mean'].values[0], norm[norm['domain'] == 'data_sparse']['std'].values[0]
        print(mean_MC, std_MC, mean_data, std_data)
    else:
        mean_MC = None
        mean_data = None
        std_MC = None
        std_data = None

    if net_type == 'classifier':
        predict=tu.predict
    elif net_type=='dann':
        predict = tud.predict
    elif net_type =='MCD':
        predict = tumcd.predict
        
    prd = predict(model,
                  df_test_prime,
                  datatype,
                  batch_size=batch_size,
                  num_workers=num_workers,
                  q=q,
                  mean=mean_MC,
                  std=std_MC)

    prd_df = pd.DataFrame({'event':prd[1], 'predicted':prd[0].flatten()})
    if len(prd)==3:
        prd_df.loc[:, 'predicted2']=prd[2].flatten()
    df_test=df_test.merge(prd_df, on='event')
    df_test.to_csv(f'{folder}/MC_6206_prediction_{format_name}.csv', index=False)

    if not sidebands_features:
        return None
    print('predicting on sidebands')
    if net_type == 'classifier':
        predict_features=tu.predict_features
    elif net_type=='dann':
        return None
    elif net_type =='MCD':
        predict_features = tumcd.predict_features


    # selection_df = pd.read_csv(sidebands_data)
    # selection_df.event = selection_df.event.astype(int)
    # selection_df.run_number = selection_df.run_number.astype(int)

    # MC_df_below = selection_df[(selection_df.domain=='MC') & (selection_df.sideband=='lower')]
    # MC_df_below = MC_df_below.merge(df_test, on = ['event', 'run_number'], how='inner').reset_index(drop=True)
    # MC_df_above = selection_df[(selection_df.domain=='MC') & (selection_df.sideband=='higher')]
    # MC_df_above = MC_df_above.merge(df_test, on = ['event', 'run_number'], how='inner').reset_index(drop=True)

    # #print(f'MC above {len(MC_df_above)}; MC below {len(MC_df_below)}'}
    # prd_feat = predict_features(model,
    #                             MC_df_below,
    #                             datatype,
    #                             batch_size=batch_size,
    #                             num_workers=num_workers,
    #                             q=q,
    #                             mean=mean_MC,
    #                             std=std_MC,
    #                             augmentation=False)
    
    # np.savez(f'{folder}/MC_6206_lowersideband_{format_name}.npz', prediction = prd_feat[0], evs = prd_feat[1], features = np.array(prd_feat[2]))

    # prd_feat = predict_features(model,
    #                             MC_df_above,
    #                             datatype,
    #                             batch_size=batch_size,
    #                             num_workers=num_workers,
    #                             q=q,
    #                             mean=mean_MC,
    #                             std=std_MC,
    #                             augmentation=False)
    
    # np.savez(f'{folder}/MC_6206_uppersideband_{format_name}.npz', prediction = prd_feat[0], evs = prd_feat[1], features = np.array(prd_feat[2]))
    

    prd_feat = predict_features(model,
                                df_test_prime,
                                datatype,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                q=q,
                                mean=mean_MC,
                                std=std_MC,
                                augmentation=False)
    
    np.savez(f'{folder}/MC_6206_features_{format_name}.npz', prediction = prd_feat[0], evs = prd_feat[1], features = np.array(prd_feat[2]))
        
    
    print ('MC written')




def pred_data(model, net_type, datatype, q, batch_size, num_workers, folder, format_name, normalize, runs=[7470, 7471, 7472, 7473], sidebands_features=False):
    df_train  = pd.read_hdf(dataset, key='data_train')
    df_test  = pd.read_hdf(dataset, key='data_valid')
    df_data_all = pd.concat([df_train, df_test], ignore_index=True)
    df_data_all.event = df_data_all.event.astype(int)
    df_data_all.run_number = df_data_all.run_number.astype(int)

    if normalize:
        norm_above = pd.read_csv(norms_above, index_col=False)[['mean', 'std', 'domain']]
        norm_below = pd.read_csv(norms_below, index_col=False)[['mean', 'std', 'domain']]
        norm = norm_above.merge(norm_below, on='domain', suffixes=['_a', '_b'])
        norm.loc[:, 'mean'] = 0.5*(norm['mean_a'] + norm['mean_b'])
        norm.loc[:, 'std' ] = 0.5*(norm['std_a' ] + norm['std_b'])
        if data_type=='dense':
            mean_MC, std_MC =  norm[norm['domain'] == 'MC_dense']['mean'].values[0], norm[norm['domain'] == 'MC_dense']['std'].values[0]
            mean_data, std_data =  norm[norm['domain'] == 'data_dense']['mean'].values[0], norm[norm['domain'] == 'data_dense']['std'].values[0]
        elif data_type=='sparse':
            mean_MC, std_MC =  norm[norm['domain'] == 'MC_sparse']['mean'].values[0], norm[norm['domain'] == 'MC_sparse']['std'].values[0]
            mean_data, std_data =  norm[norm['domain'] == 'data_sparse']['mean'].values[0], norm[norm['domain'] == 'data_sparse']['std'].values[0]
        print(mean_MC, std_MC, mean_data, std_data)
    else:
        mean_MC = None
        mean_data = None
        std_MC = None
        std_data = None

    if net_type == 'classifier':
        predict=tu.predict
    elif net_type=='dann':
        predict = tud.predict
    elif net_type =='MCD':
        predict = tumcd.predict

    for run in runs:
        df_data = df_data_all[df_data_all.run_number == run]
        prd_data = predict(model,
                           df_data,
                           datatype,
                           batch_size = batch_size,
                           num_workers = num_workers, 
                           q=q,
                           mean=mean_data,
                           std=std_data)
        prd_df = pd.DataFrame({'event':prd_data[1], 'predicted':prd_data[0].flatten()})
        if len(prd_data)==3:
            prd_df.loc[:, 'predicted2']=prd_data[2].flatten()
        df_data.event= df_data.event.astype('int')
        prd_df.event = prd_df.event.astype('int')
        df_data=df_data.merge(prd_df, on='event')
        df_data.to_csv(f"{folder}/data_{run}_prediction_{format_name}.csv", index=False) 
        print(f'data {run} written')
    
    if not sidebands_features:
        return None

    if net_type == 'classifier':
        predict_features=tu.predict_features
    elif net_type=='dann':
        return None
    elif net_type =='MCD':
        predict_features = tumcd.predict_features


    prd_feat = predict_features(model,
                                df_data_all,
                                datatype,
                                batch_size = batch_size,
                                num_workers = num_workers, 
                                q=q,
                                mean=mean_data,
                                std=std_data)
    
    np.savez(f'{folder}/data_features_{format_name}.npz', prediction = prd_feat[0], evs = prd_feat[1], features = np.array(prd_feat[2]))



    # selection_df = pd.read_csv(sidebands_data)
    # selection_df.event = selection_df.event.astype(int)
    # selection_df.run_number = selection_df.run_number.astype(int)

    # data_df_below = selection_df[(selection_df.domain=='data') & (selection_df.sideband=='lower')]
    # data_df_below = data_df_below.merge(df_data_all, on = ['event', 'run_number'], how='inner').reset_index(drop=True)
    # data_df_above = selection_df[(selection_df.domain=='data') & (selection_df.sideband=='higher')]
    # data_df_above = data_df_above.merge(df_data_all, on = ['event', 'run_number'], how='inner').reset_index(drop=True)


    # prd_feat = predict_features(model,
    #                             data_df_below,
    #                             datatype,
    #                             batch_size = batch_size,
    #                             num_workers = num_workers, 
    #                             q=q,
    #                             mean=mean_data,
    #                             std=std_data)
    
    # np.savez(f'{folder}/data_lowersideband_{format_name}.npz', prediction = prd_feat[0], evs = prd_feat[1], features = np.array(prd_feat[2]))
        

    # prd_feat = predict_features(model,
    #                             data_df_above,
    #                             datatype,
    #                             batch_size = batch_size,
    #                             num_workers = num_workers, 
    #                             q=q,
    #                             mean=mean_data,
    #                             std=std_data)
    
    # np.savez(f'{folder}/data_uppersideband_{format_name}.npz', prediction = prd_feat[0], evs = prd_feat[1], features = np.array(prd_feat[2]))
        



if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    parser = ArgumentParser(description="parameters for models")
    parser.add_argument("-conf", dest = "confname", required=True,
                        help = "input file with parameters", metavar="FILE",
                        type = lambda x: (x, is_valid_conf_file(parser, x)))
    parser.add_argument("-a", dest = "action" , required = True,
                        help = "action to do for NN",
                        type = lambda x : is_valid_action(parser, x))
    args   = parser.parse_args()
    conf_param_name = (args.confname[0].split('/')[-1]).split('.')[0]
    params = args.confname[1]
    action = args.action

    data_type = params['data_type']
    net_type = params['net_type']
    if data_type not in DataTypes:
        raise KeyError('data_type not understood')
    #construct model
    if net_type == 'classifier':
        if data_type == 'sparse':
            from src.models.nets import SparseClassifier as clf
        elif data_type == 'dense':
            from src.models.nets import DenseClassifier as clf 
        model = clf(params['mom'])
        model.cuda()
        print(model)

    elif net_type == 'dann':
        if data_type == 'sparse':
            from src.models.nets import SparseDANN as dann
        elif data_type == 'dense':
            from src.models.nets import DenseDANN as dann
        model = dann(params['mom'])
        model.cuda()


    elif net_type == 'MCD':
        from src.models.nets import Unique_Classifier as uq
        if data_type == 'sparse':
            from src.models import ResNet_sparse as rn
        elif data_type == 'dense':
            from src.models import ResNet_dense as rn   
        G = rn.Feature_extr(mom=params['mom'])
        C1 = uq(G.n_final_filters)
        C2 = uq(G.n_final_filters)
        G.cuda()
        C1.cuda()
        C2.cuda()



    weights = params['saved_weights']
    if weights:
        if net_type!='MCD':
            model.load_state_dict(torch.load(weights)['state_dict'])
            print('weights loaded')
        elif net_type=='MCD':
            G.load_state_dict(torch.load(weights)['state_dict_G'])
            C1.load_state_dict(torch.load(weights)['state_dict_C1'])
            C2.load_state_dict(torch.load(weights)['state_dict_C2'])
            model=(G, C1, C2)
    if action == 'predict':
        folder = params['predict_folder']
        format_name = params['predict_name']
        pred_MC(model, net_type, data_type, params['q_cut'], params['batch_size'], params['num_workers'], folder, format_name, params['normalize'], sidebands_features=params['sidebands_features'])
        pred_data(model, net_type, data_type, params['q_cut'], params['batch_size'], params['num_workers'], folder, format_name, params['normalize'], runs=[7470, 7471, 7472, 7473], sidebands_features=params['sidebands_features'])

    
    elif action == 'train':
        if net_type == 'classifier':
            folder_name = params['train_folder']
            train_classifier(model, data_type, params['lr'], params['n_epochs'], params['num_iter'], params['batch_size'], folder_name, params['q_cut'], params['num_workers'], params['augmentation'], args.confname[0], params['normalize'], params['one_track'], params['fiducial'])
        elif net_type == 'dann':
            folder_name = params['train_folder']
            train_dann(model, data_type, params['lr'], params['n_epochs'], params['batch_size'], folder_name, params['q_cut'], params['num_workers'])
        elif net_type=='MCD':
            folder_name = params['train_folder']
            train_mcd(G, C1, C2 , data_type, params['lr'], params['n_epochs'], params['num_iter'], params['batch_size'], folder_name, params['q_cut'], params['num_workers'], params['augmentation'], args.confname[0], params['normalize'], params['one_track'], params['fiducial'], params['n_k'])
