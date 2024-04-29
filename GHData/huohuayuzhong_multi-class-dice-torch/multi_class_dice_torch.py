import torch


def multi_class_dice(pred, mask, fp_weight=1.0, label_smooth=1.0, eps=1e-6,
                     class_weight=None, ignore_index=None, per_instance=False):
    '''
    Multi class dice
    :param pred: [n_batch, n_class, ...] 1D to 3D inputs ranged in [0, 1] (as prob)
    :param mask: [n_batch, n_class, ...] 1D to 3D inputs as a 0/1 mask
    :param fp_weight: float [0,1], penalty for fp preds, may work in data with heavy fg/bg imbalances
    :param label_smooth: float (0, inf), power of the denominator
    :param eps: epsilon, avoiding zero-division
    :param class_weight: list [float], weights for classes
    :param ignore_index: int [0, n_class), num of classes to ignore; or list [int], indexes to ignore
    :param per_instance: boolean, if True, dice was calculated per instance instead of per batch
    :return: dice score.
    '''
    nd = len(pred.shape)
    n, c, *_ = pred.shape
    # assert nd in (3, 4, 5), 'Only support 3d to 5d tensors, got {}'.format(pred.shape)
    # assert pred.shape == mask.shape, 'Sizes of inputs do not match'
    i_d = 'ncijk'[:nd]
    o_d = 'nc' if per_instance else 'c'

    intersect = torch.einsum('{i_d}, {i_d} -> {o_d}'.format(i_d=i_d, o_d=o_d), pred, mask)
    sum_ = pred + mask if fp_weight == label_smooth == 1 else fp_weight * pred ** label_smooth + mask ** label_smooth
    union = torch.einsum('{i_d} -> {o_d}'.format(i_d=i_d, o_d=o_d), sum_)
    dice_ = (2 * intersect + eps) / (union + eps)

    if class_weight is None:
        dice = dice_
    else:
        # assert isinstance(class_weight, (list, tuple, torch.FloatTensor))
        if isinstance(class_weight, (list, tuple)):
            class_weight = torch.tensor(class_weight, dtype=torch.float32).to(dice_.device)
        if torch.max(class_weight) > 1:
            class_weight /= torch.sum(class_weight)
        dice = torch.einsum('...c, c -> c', dice_, class_weight)
    if ignore_index is None:
        return torch.mean(dice)
    else:
        # assert isinstance(ignore_index, (int, list, tuple))
        if isinstance(ignore_index, int):
            return torch.mean(dice[..., ignore_index:])
        else:
            select_dim = len(dice.shape) - 1
            select_index = [i for i in range(c) if i not in ignore_index]
            return torch.mean(torch.index_select(dice, select_dim, torch.tensor(select_index).to(dice.device)))



if __name__ == '__main__':
    import numpy as np
    from medpy.metric.binary import dc
    import SimpleITK as sitk
    # pred = (np.random.uniform(0, 1, (4, 10, 512, 512)) > 0.5).astype(np.float)
    # mask = (np.random.uniform(0, 1, (4, 10, 512, 512)) > 0.5).astype(np.float)
    pred = sitk.GetArrayFromImage(sitk.ReadImage('data/src.nii.gz'))[np.newaxis]
    mask = sitk.GetArrayFromImage(sitk.ReadImage('data/w_s.nii.gz'))[np.newaxis]
    pred = (pred > 0.5).astype(np.float)
    mask = (mask > 0.5).astype(np.float)

    class_dice = []
    for i in range(mask.shape[1]):
        if np.all(pred[:, i] + mask[:, i] == 0):
            class_dice.append(1)
        else:
            class_dice.append(dc(pred[:, i], mask[:, i]))
    # print(class_dice)

    print(np.mean(class_dice))
    print(np.mean(class_dice[10:]))
    pred, mask = torch.from_numpy(pred).type(torch.float32), torch.from_numpy(mask).type(torch.float32)
    pred.requires_grad = True
    print(multi_class_dice(pred, mask))
    print(multi_class_dice(pred, mask, ignore_index=10))
    print(multi_class_dice(pred, mask, ignore_index=[i for i in range(10)]))
    print(multi_class_dice(pred, mask, fp_weight=0.5))
    print(multi_class_dice(pred, mask, label_smooth=2))
    print(multi_class_dice(pred, mask, class_weight=[i for i in range(mask.shape[1])]))
    pred.cuda(), mask.cuda()
    print(multi_class_dice(pred, mask, ignore_index=[i for i in range(10)]))
