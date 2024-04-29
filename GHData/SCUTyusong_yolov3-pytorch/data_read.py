import numpy as np
from insects import *
from image_treatment import *


# 获取一个批次内样本随机缩放的尺寸
def get_img_size(mode):  # 此函数的作用是将训练集或验证集中的图片尺寸随机缩放
    if (mode == 'train') or (mode == 'valid'):
        inds = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        ii = np.random.choice(inds)
        img_size = 320 + ii * 32
    else:
        img_size = 608
    return img_size


# a=get_img_size('train')
# print(a)


# 将 list形式的batch数据 转化成多个array构成的tuple
def make_array(batch_data):  # 将batch_data中的数据分别取出
    img_array = np.array([item[0] for item in batch_data], dtype='float32')
    gt_box_array = np.array([item[1] for item in batch_data], dtype='float32')
    gt_labels_array = np.array([item[2] for item in batch_data], dtype='int32')
    img_scale = np.array([item[3] for item in batch_data], dtype='int32')
    return img_array, gt_box_array, gt_labels_array, img_scale


# 批量读取数据，同一批次内图像的尺寸大小必须是一样的，
# 不同批次之间的大小是随机的，
# 由上面定义的get_img_size函数产生
def data_loader(datadir, batch_size=10, mode='train'):
    cname2cid = get_insect_names()
    records = get_annotations(cname2cid, datadir)

    def reader():
        if mode == 'train':
            np.random.shuffle(records)  # 将records中的record顺序打乱
        batch_data = []
        img_size = get_img_size(mode)  # 获取一个批次内样本随机缩放的尺寸
        for record in records:
            # print(record)
            img, gt_bbox, gt_labels, im_shape = get_img_data(record,
                                                             size=img_size)
            batch_data.append((img, gt_bbox, gt_labels, im_shape))
            if len(batch_data) == batch_size:
                yield make_array(batch_data)
                batch_data = []
                img_size = get_img_size(mode)
        if len(batch_data) > 0:
            yield make_array(batch_data)

    return reader


# d = data_loader('./insects/train', batch_size=2, mode='train')
# img, gt_boxes, gt_labels, im_shape = next(d())
# print(img.shape, gt_boxes.shape, gt_labels.shape, im_shape)
# 测试数据读取

# 将 list形式的batch数据 转化成多个array构成的tuple
def make_test_array(batch_data):
    img_name_array = np.array([item[0] for item in batch_data])
    img_data_array = np.array([item[1] for item in batch_data], dtype = 'float32')
    img_scale_array = np.array([item[2] for item in batch_data], dtype='int32')
    return img_name_array, img_data_array, img_scale_array

# 测试数据读取
def test_data_loader(datadir, batch_size= 10, test_image_size=608, mode='test'):
    """
    加载测试用的图片，测试数据没有groundtruth标签
    """
    image_names = os.listdir(datadir)
    def reader():
        batch_data = []
        img_size = test_image_size
        for image_name in image_names:
            file_path = os.path.join(datadir, image_name)
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            H = img.shape[0]
            W = img.shape[1]
            img = cv2.resize(img, (img_size, img_size))

            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            mean = np.array(mean).reshape((1, 1, -1))
            std = np.array(std).reshape((1, 1, -1))
            out_img = (img / 255.0 - mean) / std
            out_img = out_img.astype('float32').transpose((2, 0, 1))
            img = out_img #np.transpose(out_img, (2,0,1))
            im_shape = [H, W]

            batch_data.append((image_name.split('.')[0], img, im_shape))
            if len(batch_data) == batch_size:
                yield make_test_array(batch_data)
                batch_data = []
        if len(batch_data) > 0:
            yield make_test_array(batch_data)

    return reader
# d = test_data_loader('./insects/test/images',mode='test')
# img, gt_boxes, gt_labels = next(d())
# print(img.shape, gt_boxes.shape, gt_labels.shape)