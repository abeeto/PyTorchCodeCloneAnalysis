import os
import sys

import numpy as np

neglected_paths = ['/usr/local/Ascend/nnae/latest/fwkacllib/python/site-packages', '/usr/local/Ascend/tfplugin/latest/tfplugin/python/site-packages', '/home/ma-user/anaconda3/envs/MindSpore/lib/python3.7', '/home/ma-user/anaconda3/envs/MindSpore/lib/python3.7/site-packages', '/home/ma-user/modelarts/modelarts-sdk', '/opt/conda/lib/python3.7/site-packages']
# neglected_paths = ['/usr/local/Ascend/nnae/latest/fwkacllib/python/site-packages', '/usr/local/Ascend/nnae/latest/fwkacllib/python/site-packages/auto_tune.egg', '/usr/local/Ascend/nnae/latest/fwkacllib/python/site-packages/schedule_search.egg', '/usr/local/Ascend/tfplugin/latest/tfplugin/python/site-packages', '/usr/local/Ascend/nnae/latest/opp/op_impl/built-in/ai_core/tbe',
#                    '/home/ma-user/anaconda3/envs/MindSpore/lib/python37.zip', '/home/ma-user/anaconda3/envs/MindSpore/lib/python3.7', '/home/ma-user/anaconda3/envs/MindSpore/lib/python3.7/lib-dynload', '/home/ma-user/anaconda3/envs/MindSpore/lib/python3.7/site-packages', '/home/ma-user/modelarts/modelarts-sdk', '/home/ma-user/modelarts/common-algo-toolkit', '/home/ma-user/modelarts/ma-cli',
#                    '/opt/conda/lib/python3.7/site-packages']
for path in neglected_paths:
    sys.path.append(path)

import mindspore
import mindspore.dataset as ds

import mindspore.common.dtype as mstype
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.vision.py_transforms as P
from PIL import Image
from mindspore import context
from mindspore import Tensor
import mindspore.ops as ops


class imgdataset():
    def __init__(self, dataset_dir, txt_path, transformer='train'):
        with open(txt_path) as f:
            line = f.readlines()
            self.img_list = [os.path.join(dataset_dir, i.split()[0]) for i in line]
            self.label_list = [int(i.split()[1]) for i in line]
            self.cam_list = [int(i.split()[2]) for i in line]

    # self.cam_list = [int(i.split('c')[1][0]) for i in line]

    def __getitem__(self, index):
        im_path = self.img_list[index]
        image = Image.open(im_path)
        return image, self.label_list[index], self.cam_list[index]

    def __len__(self):
        return len(self.label_list)


class imgdataset_camtrans():
    def __init__(self, dataset_dir, txt_path, num_cam, K):
        self.num_cam = num_cam
        self.K = K
        with open(txt_path) as f:
            line = f.readlines()
            self.img_list = [os.path.join(dataset_dir, i.split()[0]) for i in line]
            self.label_list = [int(i.split()[1]) for i in line]
            self.cam_list = [int(i.split()[2]) for i in line]

    def __getitem__(self, index):
        im_path = self.img_list[index]
        camid = self.cam_list[index]
        # self.num_cam_ = Tensor([self.num_cam], dtype=mindspore.int32)

        # cams = ops.Randperm(max_length=self.num_cam, pad=-1)(Tensor([self.num_cam], dtype=mstype.int32))
        # cams_ = Tensor(cams, dtype=mindspore.int32) + 1
        cams = np.random.permutation(self.num_cam) + 1

        imgs = []
        # print("cams[0:self.K]:", cams[0:self.K])
        for sel_cam in cams[0:self.K]:  # todo might be wrong
            if sel_cam != camid:
                im_path_cam = im_path[:-4] + '_fake_' + str(camid) + 'to' + str(sel_cam) + '.jpg'
            else:
                im_path_cam = im_path
            image = Image.open(im_path_cam)
            imgs.append(image)
        # print("len(imgs):", len(imgs))
        if len(imgs) == 8:
            return imgs[0], imgs[1], imgs[2], imgs[3], imgs[4], imgs[5], imgs[6], imgs[7], self.label_list[index], self.cam_list[index]

        return imgs[0], imgs[1], imgs[2], imgs[3], imgs[4], imgs[5], self.label_list[index], index  # todo: it's not elegant. But I don't know how to do it better.
        # return (*imgs, self.label_list[index], index)

    def __len__(self):
        return len(self.label_list)


def create_dataset(dataset_dir, ann_file, batch_size, state, num_cam=6, K=6, num_workers=1):
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    train_transform = [
        C.Resize((256, 128)),
        # C.RandomHorizontalFlip(), # todo: remove randomness
        C.Normalize(mean=mean, std=std),
        # P.RandomErasing(prob=0.5),# todo: remove randomness
        C.HWC2CHW(),
    ]

    test_transform = [
        C.Resize((256, 128)),
        C.Normalize(mean=mean, std=std),
        C.HWC2CHW(),
    ]
    if state == 'train':
        transform = train_transform
        columns_names_list = ['images' + str(i) for i in range(num_cam)]
        columns_names_list.append('labels')
        columns_names_list.append('index')
    elif state == 'test':
        transform = test_transform
        columns_names_list = ["images", 'labels', 'index']

    # columns_names_list.append('im_path')  # todo: delete im_path

    if state == 'train':
        dataset_generator = imgdataset_camtrans(dataset_dir=dataset_dir, txt_path=ann_file, num_cam=num_cam, K=K)
        dataset = ds.GeneratorDataset(dataset_generator, column_names=columns_names_list, num_parallel_workers=num_workers, shuffle=False)
        for i in range(num_cam):
            dataset = dataset.map(operations=transform, input_columns=columns_names_list[i])
    elif state == 'test':
        dataset_generator = imgdataset(dataset_dir=dataset_dir, txt_path=ann_file)
        dataset = ds.GeneratorDataset(dataset_generator, column_names=columns_names_list, num_parallel_workers=num_workers, shuffle=False)
        dataset = dataset.map(operations=transform, input_columns=columns_names_list[0])

    dataset = dataset.batch(batch_size=batch_size, drop_remainder=False)
    dataset = dataset.repeat(1)
    return dataset


# def create_dataset(dataset_path, do_train, repeat_num=1, batch_size=32, target="Ascend", distribute=False):
#     # device number: total number of devices of training
#     # rank_id: the sequence of current device of training
#     # device_num, rank_id = _get_rank_info()
#     # if distribute:
#     #     init()
#     #     rank_id = get_rank()
#     #     device_num = get_group_size()
#     # else:
#     #     device_num = 1
#     device_num = 1
#     if device_num == 1:
#         # standalone training
#         # num_paralel_workers: parallel degree of data process
#         # shuffle: whether shuffle data or not
#         data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=8, shuffle=True)
#     else:
#         # distributing traing (meaning of num_parallel_workers and shuffle is same as above)
#         # num_shards: total number devices for distribute training, which equals number shard of data
#         # shard_id: the sequence of current device in all distribute training devices, which equals the data shard sequence for current device
#         data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=8, shuffle=True, num_shards=device_num, shard_id=rank)
#     # define data operations
#     trans = []
#     if do_train:
#         trans += [
#             C.RandomHorizontalFlip(prob=0.5)
#         ]
#     trans += [
#         C.Resize((256, 256)),
#         C.CenterCrop(224),
#         C.Rescale(1.0 / 255.0, 0.0),
#         C.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#         C.HWC2CHW()
#     ]
#     type_cast_op = C2.TypeCast(mstype.int32)
#     # call data operations by map
#     data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=8)
#     data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=8)
#     # batchinng data
#     data_set = data_set.batch(batch_size, drop_remainder=True)
#     # repeat data, usually repeat_num equals epoch_size
#     data_set = data_set.repeat(repeat_num)
#     return data_set

if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=0)
    dataset_path = 'data'
    ann_file_train = 'list_duke/list_duke_train.txt'
    ann_file_test = 'list_duke/list_duke_test.txt'

    snapshot = 'ms_m2d.ckpt'  # todo

    num_cam = 8
    K = 8

    train_dataset_path = dataset_path + '/duke_merge'
    test_dataset_path = dataset_path + '/duke'

    train_dataset = create_dataset(train_dataset_path, ann_file_train, batch_size=1, state='train', num_cam=num_cam, K=num_cam, )
    test_datset = create_dataset(test_dataset_path, ann_file_test, batch_size=1, state='test')

    img_d = imgdataset_camtrans(dataset_dir=train_dataset_path, txt_path=ann_file_train, num_cam=num_cam, K=K)
    for i in range(10):
        print("########################################")
        dd = img_d.__getitem__(i)

    ##########nhuk#################################### test train_dataset
    iter_n = 0
    for data in train_dataset.create_dict_iterator():
        print("######################################## train_dataset_dict_iterator")
        print("data.keys()", data.keys())
        data1 = data['images0'].asnumpy()
        label1 = data['labels'].asnumpy()
        print(data1.shape)
        iter_n += 1
        if iter_n == 10:
            break
    ##########nhuk####################################

    # ##########nhuk#################################### test test_dataset
    # for data in test_datset.create_dict_iterator():
    #     data1 = data['images'].asnumpy()
    #     label1 = data['labels'].asnumpy()
    #     print(data1.shape)
    #     print(type(data1))
    #     whimsy.get_statistics(data1)
    #     time.sleep(4)
    # ##########nhuk####################################

    # for data1, label1, _ in train_dataset.create_tuple_iterator():
    #     # data1 = data['images'].asnumpy()
    #     # label1 = data['label'].asnumpy()
    #     print("IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII 5")
    #     # print(f'data:[{data1[0]:7.5f}, {data1[1]:7.5f}], label:[{label1[0]:7.5f}]')
    #     print(data1.shape)
    #     exit()

# import mindspore.ops as ops
# from mindspore import Tensor
# from mindspore import dtype as mstype
# # MindSpore
# # The result of every execution is different because this operator will generate n random samples.
# num_cam =3
# randperm = ops.Randperm(max_length=num_cam, pad=-1)
# n = Tensor([num_cam], dtype=mstype.int32)
# output = randperm(n)
# print(output)
# # Out:
# # [15 6 11 19 14 16 9 5 13 18 4 10 8 0 17 2 1 12 3 7
# #  -1 -1 -1 -1 -1 -1 -1 -1 -1 -1]
