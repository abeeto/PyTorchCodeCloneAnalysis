import torch
from torch.functional import F

# pip install torch-scatter

import torch
import torch.nn as nn
from torch_scatter import scatter


def discriminative_loss_single(prediction, correct_label, delta_v=0.5, delta_d=1.5, param_var=1.0, param_dist=1.0,
                               param_reg=0.001, back_val=None,
                               plt_array=None,plt_skip_zero_point=False):
    """

    :param prediction: D x MN. D is the dimension of the embedded. MN is the imaege
    :param correct_label: MN the image size
    :param delta_v:
    :param delta_d:
    :param feature_dim:
    :param param_var:
    :param param_dist:
    :param param_reg:
    :return:
    """

    feature_dim = prediction.shape[0]
    correct_label = correct_label.view(1, -1)

    reshaped_pred = prediction.view(feature_dim, -1)
    unique_labels, unique_ids = torch.unique(correct_label, sorted=True, return_inverse=True)

    if plt_array is not None:
        # draw the points
        for i in range(feature_dim - 1):
            if len(plt_array) <= i:
                break
            if plt_skip_zero_point:
                mask_out_zero = correct_label[0] > 0
            else:
                mask_out_zero = correct_label[0] >-123
            plt_array[i].scatter(x=reshaped_pred[i][mask_out_zero].cpu().detach().numpy(),
                                 y=reshaped_pred[i + 1][mask_out_zero].cpu().detach().numpy(),
                                 c=correct_label[0][mask_out_zero].cpu().detach().numpy(), linewidths=1, s=1)

    if torch.cuda.is_available():
        unique_labels = unique_labels.cuda().type(torch.cuda.LongTensor)
        unique_ids = unique_ids.cuda().type(torch.cuda.LongTensor)
    num_instances = unique_labels.size()[0]

    # the mean of embedding of clusters
    segment_mean = torch.zeros((feature_dim, num_instances), dtype=torch.float32)
    if torch.cuda.is_available():
        segment_mean = segment_mean.cuda()
    for i, lb in enumerate(unique_labels):
        mask = correct_label.eq(lb).repeat(feature_dim, 1)
        segment_embedding = torch.masked_select(reshaped_pred, mask).view(feature_dim, -1)
        if (back_val is not None) and (len(back_val)>i):  #
            segment_mean[:, i] = back_val[i]
        else:
            segment_mean[:, i] = torch.mean(segment_embedding, dim=1)

    unique_ids = unique_ids.view(-1)
    mu_expand = segment_mean.index_select(1, unique_ids)
    # print("mu_expand", mu_expand.size())
    distance = mu_expand - reshaped_pred
    distance = distance.norm(2, 0, keepdim=True)
    distance = distance - delta_v
    distance = F.relu(distance)
    distance = distance ** 2

    l_var = torch.empty(num_instances, dtype=torch.float32)
    if torch.cuda.is_available():
        l_var = l_var.cuda()
    for i, lb in enumerate(unique_labels):
        mask = correct_label.eq(lb)
        var_sum = torch.masked_select(distance, mask)
        l_var[i] = torch.mean(var_sum)

    l_var = torch.mean(l_var)

    seg_interleave = segment_mean.permute(1, 0).repeat(num_instances, 1)
    seg_band = segment_mean.permute(1, 0).repeat(1, num_instances).view(-1, feature_dim)

    dist_diff = seg_interleave - seg_band
    mask = (1 - torch.eye(num_instances, dtype=torch.int8)).view(-1, 1).repeat(1, feature_dim)
    if torch.cuda.is_available():
        mask = mask.cuda().type(torch.cuda.BoolTensor)  # torch.cuda.ByteTensor)
    dist_diff = torch.masked_select(dist_diff, mask).view(-1, feature_dim)
    dist_norm = dist_diff.norm(2, 1)
    dist_norm = 2 * delta_d - dist_norm
    dist_norm = F.relu(dist_norm)
    dist_norm = dist_norm ** 2
    l_dist = torch.mean(dist_norm)
    # 正则化项
    l_reg = torch.mean(torch.norm(segment_mean, 2, 0))

    l_var = param_var * l_var
    l_dist = param_dist * l_dist
    l_reg = param_reg * l_reg
    loss = l_var + l_dist + l_reg

    return loss


class DiscriminativeLoss(nn.Module):
    def __init__(self):
        super(DiscriminativeLoss, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self, binary_logits, binary_labels,
                instance_logits, instance_labels, delta_v=0.5, delta_d=3, plt_bin=None, plt_inst=None, plt_skip_zero_point=False):
        """

        :param binary_logits:[batch_size=2, embedding_size=2, imagesizeMN=(256x512)]
        :param binary_labels:[batch_size=2, imagesizeMN=(256x512)]
        :param instance_logits:[batch_size=2, embedding_size=5, imagesizeMN=(256x512)]
        :param instance_labels:[batch_size=2, imagesizeMN=(256x512)]
        :param delta_v:
        :param delta_d:
        :param plt_bin:
        :param plt_inst:
        :return:
        """
        instance_segmenatation_loss = torch.tensor(0.).cuda()
        bin_segmenatation_loss = torch.tensor(0.).cuda()
        batch_size = instance_logits.shape[0]
        for dimen in range(batch_size):
            # instance_loss = loss_set[dimen]

            # prediction = torch.unsqueeze(instance_logits[dimen], 0).cuda()
            # correct_label = torch.unsqueeze(instance_labels[dimen], 0).cuda()
            prediction = instance_logits[dimen]
            correct_label = instance_labels[dimen]
            # instance_segmenatation_loss += discriminative_loss_single(prediction, correct_label, delta_v, delta_d, back_val=torch.tensor([1,0,0,0,0]).cuda(), plt_array=plt_inst)
            instance_segmenatation_loss += discriminative_loss_single(prediction, correct_label, delta_v, delta_d,
                                                                      back_val=None,
                                                                      plt_array=plt_inst,plt_skip_zero_point=plt_skip_zero_point)

            bin_segmenatation_loss += discriminative_loss_single(binary_logits[dimen], binary_labels[dimen], delta_v,
                                                                 delta_d, back_val=None, plt_array=plt_bin, plt_skip_zero_point=plt_skip_zero_point)

        bin_segmenatation_loss = bin_segmenatation_loss / batch_size
        instance_segmenatation_loss = instance_segmenatation_loss / batch_size
        return bin_segmenatation_loss, instance_segmenatation_loss
