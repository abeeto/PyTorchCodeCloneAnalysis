import time

import mindspore
import numpy as np
from mindspore import context
from scipy.spatial.distance import cdist

from dataset import create_dataset
from evaluate_joint_sim import evaluate_joint
from resnet import ResNet, load_ms_resnet50_model, Bottleneck
from st_distribution import get_st_distribution
from utils import extract_fea_camtrans, get_info, extract_fea_test, cluster

if __name__ == '__main__':
    mindspore.set_seed(0)
    start_time = time.time()

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

    dataset_path = 'data'
    ann_file_train = 'list_duke/list_duke_train.txt'
    ann_file_test = 'list_duke/list_duke_test.txt'

    snapshot = 'ms_m2d.ckpt'  # todo

    num_cam = 8
    K = 8

    ##########nhuk#################################### dataset setting
    train_dataset_path = dataset_path + '/duke_merge'
    test_dataset_path = dataset_path + '/duke'
    train_dataset = create_dataset(dataset_dir=train_dataset_path, ann_file=ann_file_train, batch_size=1, state='train', num_cam=num_cam, K=K)
    test_dataset = create_dataset(dataset_dir=test_dataset_path, ann_file=ann_file_test, state='test', batch_size=1)
    ##########nhuk####################################

    model = ResNet(Bottleneck, [3, 4, 6, 3], 751, train=False)
    model = load_ms_resnet50_model(model, snapshot)
    model.set_train(False)

    print('extract feature for training set')
    train_feas = extract_fea_camtrans(model, train_dataset, num_cam, K)

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
