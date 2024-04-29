import os
import sys
import time

import mindspore

from evaluate_joint_sim import evaluate_joint

neglected_paths = ['/usr/local/Ascend/nnae/latest/fwkacllib/python/site-packages', '/usr/local/Ascend/nnae/latest/fwkacllib/python/site-packages/auto_tune.egg', '/usr/local/Ascend/nnae/latest/fwkacllib/python/site-packages/schedule_search.egg', '/usr/local/Ascend/tfplugin/latest/tfplugin/python/site-packages', '/usr/local/Ascend/nnae/latest/opp/op_impl/built-in/ai_core/tbe',
                   '/home/ma-user/anaconda3/envs/MindSpore/lib/python37.zip', '/home/ma-user/anaconda3/envs/MindSpore/lib/python3.7', '/home/ma-user/anaconda3/envs/MindSpore/lib/python3.7/lib-dynload', '/home/ma-user/anaconda3/envs/MindSpore/lib/python3.7/site-packages', '/home/ma-user/modelarts/modelarts-sdk', '/home/ma-user/modelarts/common-algo-toolkit', '/home/ma-user/modelarts/ma-cli',
                   '/opt/conda/lib/python3.7/site-packages']
for path in neglected_paths:
    sys.path.append(path)

os.system('export LD_PRELOAD=$LD_PRELOAD:/home/ma-user/anaconda3/envs/MindSpore/lib/python3.7/site-packages/torch/lib/libgomp-d22c30c5.so.1')

import numpy as np
from mindspore import context
from scipy.spatial.distance import cdist

from dataset import create_dataset
from resnet import ResNet, load_ms_resnet50_model, Bottleneck
from st_distribution import get_st_distribution
from utils import extract_fea_camtrans, get_info, extract_fea_test, cluster

if __name__ == '__main__':
    ################################################## params
    mindspore.set_seed(0)
    start_time = time.time()
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

    dataset_path = 'data'

    ##########nhuk####################################
    ann_file_train = 'list_market/list_market_train.txt'
    ann_file_test = 'list_market/list_market_test.txt'
    ##########nhuk####################################

    # ##########nhuk#################################### # todo change to list_market_train.txt
    # ann_file_train = 'list_market/list_market_train_mini.txt'
    # ann_file_test = 'list_market/list_market_test_mini.txt'
    # ##########nhuk####################################

    snapshot = 'm_resnet.ckpt'

    num_cam = 6
    K = 6
    ################################################## dataset setting
    train_dataset_path = dataset_path + '/market_merge'
    test_dataset_path = dataset_path + '/Market-1501-v15.09.15/Market-1501-v15.09.15'
    train_dataset = create_dataset(dataset_dir=train_dataset_path, ann_file=ann_file_train, batch_size=1, state='train', num_cam=num_cam, K=K, num_workers=32)

    test_dataset = create_dataset(dataset_dir=test_dataset_path, ann_file=ann_file_test, state='test', batch_size=1, num_workers=32)

    # ##########nhuk#################################### test code
    # stat_counts = 100
    # with open("rubb/ms_seq_data_info.txt", "w") as f:
    #     count = 0
    #     for data in train_dataset.create_dict_iterator():
    #         # f.write(data["labels"] + " " + data["index"] + "\n")
    #         f.write(str(data["labels"]) + " " + str(data["index"]) + "\n")
    #         count += 1
    #         if count == stat_counts:
    #             break
    #     f.write("##################################################" + "\n")
    #     count = 0
    #     for data in test_dataset.create_dict_iterator():
    #         f.write(str(data["labels"]) + " " + str(data["index"]) + "\n")
    #         count += 1
    #         if count == stat_counts:
    #             break
    # exit()
    # ##########nhuk####################################

    # ##########nhuk#################################### test code
    # for i in range(10):
    #     data_item, _, _ = test_dataset.__getitem__(i)
    #     get_statistics(data_item)
    # exit()
    # ##########nhuk####################################

    model = ResNet(Bottleneck, [3, 4, 6, 3], 702, train=False)
    model = load_ms_resnet50_model(model, snapshot)
    model.set_train(False)

    print('extract feature for training set')
    train_feas = extract_fea_camtrans(model, train_dataset)

    np.save("rubb/ms_train_feas.npy", train_feas)
    _, cam_ids, frames = get_info(ann_file_train)
    print('generate spatial-temporal distribution')
    dist = cdist(train_feas, train_feas)
    np.save("rubb/ms_dist.npy", dist)
    dist = np.power(dist, 2)
    labels = cluster(dist)
    np.save("rubb/ms_labels.npy", labels)
    num_ids = len(set(labels))
    print('cluster id num:', num_ids)
    distribution = get_st_distribution(cam_ids, labels, frames, id_num=num_ids, cam_num=num_cam)
    np.save("rubb/ms_distribution.npy", distribution)

    print('extract feature for testing set')
    test_feas = extract_fea_test(model, test_dataset)
    np.save("rubb/ms_test_feas.npy", test_feas)

    print('evaluation')
    evaluate_joint(test_fea=test_feas, st_distribute=distribution, ann_file=ann_file_test, select_set='market')
    end_time = time.time()
    # convert time to mins
    print('inference time:', (end_time - start_time) / 60)
