from torch.utils.tensorboard import SummaryWriter
import os
import tqdm
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import torch
import math

def image_list_blend(org, img_list):
    base = img_list[0]
    for img in img_list:
        base = np.maximum(base, img)
    ary = np.array(base, dtype=np.float)
    plt.imsave('temp.png', ary, cmap='hot')
    temp = Image.open('temp.png')
    temp = temp.resize((64,64))
    temp = temp.convert("RGBA")
    org = Image.fromarray(org)
    org = org.resize((64,64))
    org = org.convert("RGBA")
    blended = Image.blend(org, temp, 0.45)
    return blended


def pair_joint_to_person(pair, joint, crop_size=None):
    center_count = len(joint[-1])
    joint_dict = [{} for _ in range (center_count)]

    shape = 64
    if crop_size is None:
        scale = 1
    else:
        crop_size = list(map(float, crop_size.split(',')))
        scale = np.asarray([int(crop_size[2] - crop_size[0]), int(crop_size[3] - crop_size[1])])/np.array([shape, shape])

    for i in range(15):
        for j, idxes in enumerate(pair[i]):
            if idxes == -1:
                joint_dict[j][i] = None
            else:
                joint_dict[j][i] = np.asarray(joint[i][idxes] * scale, dtype=int)
    return joint_dict


def get_joint_from_heatmap(heatmap):
    joints = []
    for r in range(1,63):
        for c in range(1,63):
            if np.max(heatmap[r-1:r+2,c-1:c+2]) == heatmap[r,c] and heatmap[r,c]>0.3:
                joints.append([c,r])
    return joints


def make_pair(joints, centers):
    pairs = []
    for center in centers:
        for joint in joints:
            pairs.append([joint, center])
    return pairs


def calc_energy(pt, center, limb):
    pt = np.asarray(pt)
    center = np.asarray(center)
    gt = draw_limb(pt, center, 64)
    distance = point_distance(pt, center)

    energy = np.sum(limb * gt)
    if distance == 0:
        distance = 1
    return energy/distance


def find_max_energy(energy, idx, max_idx_list):
    max_idx = np.argmax(energy[idx])
    if energy[idx][max_idx] == 0:
        max_idx_list[idx]=-1
        return
    is_duplicate = np.min(np.abs(max_idx_list - max_idx)) == 0
    if is_duplicate:
        for i, ot_idx in enumerate(max_idx_list):
            if ot_idx == max_idx:
                if energy[i][ot_idx] >= energy[idx][max_idx]:
                    energy[idx][max_idx]=0
                    find_max_energy(energy, idx, max_idx_list)
                else:
                    energy[i][ot_idx]=0
                    max_idx_list[i]=-1
                    find_max_energy(energy, i, max_idx_list)
    else:
        max_idx_list[idx] = max_idx

    return


def find_max_energy_sum(energy):
    max_idx_list = np.zeros((energy.shape[0],), dtype=np.int) - 1
    for idx in range(energy.shape[0]):
        find_max_energy(energy, idx, max_idx_list)
    return max_idx_list


def grouping_joints(joint, center, limb):
    pairs = make_pair(joint, center)
    point_count = len(joint)
    center_count = len(center)

    energy = []
    for pair in pairs:
        energy.append(calc_energy(pair[0], pair[1], limb))

    if len(energy) == 0:
        return np.zeros((center_count,)) - 1

    energy = np.reshape(np.array(energy), (center_count, -1))
    return find_max_energy_sum(energy)


def inference_joints(heatmaps, limbs):
    ch = heatmaps.shape[0]
    joints_list = [[] for _ in range(ch)]

    for idx, heatmap in enumerate(heatmaps):
        joints = get_joint_from_heatmap(heatmap)
        joints_list[idx] = joints

    count = 0
    for i in range(len(joints_list)):
        count = count + len(joints_list[i])

    print(count)

    max_pair = [[] for _ in range(ch)]
    for idx, joints in enumerate(joints_list[:-1]):
        max_pair[idx] = grouping_joints(joints, joints_list[-1], limbs[idx*2:idx*2+2])

    return max_pair, joints_list


def limb_to_show(limb_map):
    x = limb_map[0,:,:]
    y = limb_map[1,:,:]
    divider = np.abs(np.where(x==0,1,x))
    test = np.arctan(np.abs(y)/divider)
    test = test/math.pi

    check_plain_x = np.where(x > 0, 2, -2)
    check_plain_y = np.where(y > 0, 1, -1)
    check_plain = check_plain_x + check_plain_y
    check_plain = np.where(check_plain == 3, 2, check_plain)
    plain = np.where(check_plain == -3, -2, check_plain) + 2
    plain = plain*90
    test = (test*180+plain)/360
    test[0,0]=1

    # plt.imshow(test, cmap='inferno')
    # plt.show()
    return test


def point_distance(pt1, pt2):
    return np.sqrt(np.sum(np.square(pt1 - pt2)))


def draw_limb(pt, center_pt, shape):
    pt_distance = point_distance(pt, center_pt)
    distance_vector = (pt - center_pt) / pt_distance if pt_distance != 0 else np.asarray([0,0])

    base = np.zeros((shape, shape, 3), dtype=np.uint8)
    img = Image.fromarray(base)
    draw_img = ImageDraw.Draw(img)
    draw_img.line([center_pt[0], center_pt[1], pt[0], pt[1]], fill='white', width=1)
    img = img.convert('L')
    img = np.asarray(img)/255
    img = np.where(img > 0.5, 1, 0)
    base = np.asarray([np.asarray(img), np.asarray(img)])
    base = np.transpose(base, (1,2,0))

    distance_limb = np.transpose(base*distance_vector, (2, 0, 1))
    return distance_limb


def limb_merge(limb_list, shape):
    if len(limb_list) == 0:
        return np.zeros((2,64,64))
    limb_list = np.reshape(np.asarray(limb_list), (-1, 2, shape, shape))

    divider = np.where(limb_list != 0, 1, 0)
    divider = divider.sum(axis=0)
    divider = np.where(divider[0,:,:] == 0, 1, divider[0,:,:])

    # count = np.squeeze(count)
    limb_sum = limb_list.sum(axis=0)
    distance_limb = limb_sum/divider
    return distance_limb


def points_to_limb(points_list, shape, ch, crop_size=None):

    temp_limb = [[] for _ in range(ch)]
    if crop_size is None:
        scale = 1
    else:
        scale = np.array([shape, shape]) / np.asarray([crop_size[2]-crop_size[0], crop_size[3]-crop_size[1]])
    for points in points_list:
        if len(points) == 0:
            continue
        center_key = list(points.keys())[-1]
        center_pt = np.asarray(points[center_key]*scale, dtype=int)
        for pt_key in points:
            # filtering center
            if pt_key == center_key:
                continue
            pt = np.asarray(points[pt_key]*scale, dtype=int)
            limb = draw_limb(pt, center_pt, shape)
            temp_limb[pt_key].append(limb)

    limb = limb_merge(temp_limb[0], shape)

    for i in range(1, ch):
        temp = limb_merge(temp_limb[i], shape)
        limb = np.concatenate((limb,temp), axis=0)

    return limb


def create_2d_heat_map(point, size, sigma=3):
    base = np.zeros(size)

    x = math.ceil(point[0])
    y = math.ceil(point[1])

    for r in range(size[0]):
        for c in range(size[1]):
            base[r, c] = np.exp(-((r - y) ** 2 + (c - x) ** 2) / sigma)

    return base


class TrainInfo:
    def __init__(self, path):
        self.save_path = path
        self.train_loss = math.inf
        self.validate_loss = math.inf
        self.train_acc = []
        self.validate_acc = []
        self.min_train_loss = math.inf
        self.min_validate_loss = math.inf
        self.max_train_acc = None
        self.max_validate_acc = None

    def set_train_loss(self, loss):
        self.train_loss = loss

    def set_val_loss(self, loss):
        self.validate_loss = loss

    def set_train_acc(self, acc):
        self.train_acc = acc

    def set_val_acc(self, acc):
        self.validate_acc = acc


class TorchBoard:
    def __init__(self, dir_path, comment):
        self.writer = SummaryWriter(log_dir=dir_path, comment=comment)

    def add_train_loss(self, value, n_iter):
        self.writer.add_scalar('Loss/train', value, n_iter)

    def add_train_acc(self, value, n_iter, name):
        self.writer.add_scalar('{}/train'.format(name), value, n_iter)

    def add_val_loss(self, value, n_iter):
        self.writer.add_scalar('Loss/test', value, n_iter)

    def add_val_acc(self, value, n_iter, name):
        self.writer.add_scalar('{}/test'.format(name), value, n_iter)


def train_model(epoches, model, loss, optim, train_loader, val_loader, scheduler=None, save_path=None, tag=None, checkpoint=None, accuracy=None):
    def calc_loss_optim(loader, conf, accuracy, optim):
        is_train = optim is not None
        with torch.set_grad_enabled(is_train):
            if accuracy is not None:
                total_acc = [0 for _ in range(len(accuracy))]
            else:
                total_acc = None
            total_loss = 0
            for iter, (x, y) in tqdm.tqdm(enumerate(loader(conf))):
                if is_train:
                    optim.zero_grad()
                result = model(x)
                iter_loss = loss(y, result)
                total_loss += iter_loss
                if is_train:
                    iter_loss.backward()
                    optim.step()
                if accuracy is not None:
                    for idx, acc_di in enumerate(accuracy):
                        metrics = acc_di['metrics']
                        iter_acc = metrics(y, result)
                        total_acc[idx] += iter_acc
                del iter_loss
                del result
            total_loss /= iter+1
        return total_loss, total_acc
    print()
    print("{0:^40s}".format('Train Information'))
    print('{0:^40s}'.format("{0:22s}: {1:10,d}".format('model # param', get_param_count(model))))
    print("{0:^40s}".format("{0:22s}: {1:10,d}".format('epoch', epoches)))
    print("{0:^40s}".format("{0:22s}: {1:10,d}".format('batch size', train_loader['conf']['batch'])))

    train_info = TrainInfo(save_path)

    if torch.cuda.is_available():
        print('make cuda')
        model = model.cuda()
    torch_board_path = '{}\\{}'.format(save_path, tag)
    os.makedirs(torch_board_path, exist_ok=True)
    tb = TorchBoard(torch_board_path, tag)
    for epoch in range(1, epoches+1):
        model.train()
        # train_loss = 0
        # val_loss = 0
        # if accuracy is not None:
        #     train_acc = [0 for _ in range(len(accuracy))]
        # if accuracy is not None:
        #     val_acc = [0 for _ in range(len(accuracy))]
        #
        # for iter, (x, y) in tqdm.tqdm(enumerate(train_loader['loader'](train_loader['conf']))):
        #     optim.zero_grad()
        #     result = model(x)
        #     iter_loss = loss(y, result)
        #     train_loss += iter_loss
        #     iter_loss.backward()
        #     optim.step()
        #     if accuracy is not None:
        #         for idx, acc_di in enumerate(accuracy):
        #             metrics = acc_di['metrics']
        #             acc = metrics(y, result)
        #             train_acc[idx] += acc
        #     del iter_loss
        #     del result
        train_loss, train_acc = calc_loss_optim(train_loader['loader'], train_loader['conf'], accuracy, optim)
        train_info.set_train_loss(train_loss)

        if accuracy is not None:
            train_acc = np.array(train_acc)/(iter+1)
            train_info.set_train_acc(train_acc)

        model.eval()
        # with torch.no_grad():
        #     for iter, (x, y) in enumerate(val_loader['loader'](val_loader['conf'])):
        #         result = model(x)
        #         iter_loss = loss(y, result)
        #         val_loss += iter_loss
        #         if accuracy is not None:
        #             for idx, acc_di in enumerate(accuracy):
        #                 metrics = acc_di['metrics']
        #                 acc = metrics(y, result)
        #                 val_acc[idx] += acc
        #
        #         del iter_loss
        #         del result
        # val_loss /= (iter+1)
        val_loss, val_acc = calc_loss_optim(val_loader['loader'], val_loader['conf'], accuracy, None)
        train_info.set_val_loss(val_loss)
        if accuracy is not None:
            val_acc = np.array(val_acc)/(iter+1)
            train_info.set_val_acc(val_acc)

        if save_path is not None:
            tb.add_train_loss(train_loss, epoch)
            tb.add_val_loss(val_loss, epoch)

            if accuracy is not None:
                for idx, acc_di in enumerate(accuracy):
                    tb.add_val_acc(val_acc[idx], epoch, acc_di['name'])
                    tb.add_train_acc(train_acc[idx], epoch, acc_di['name'])

        if checkpoint is not None:
            checkpoint(model, train_info)

        if scheduler is not None:
            scheduler.step()


def get_param_count(net):
    total_params = sum(p.numel() for p in net.parameters())
    return total_params
