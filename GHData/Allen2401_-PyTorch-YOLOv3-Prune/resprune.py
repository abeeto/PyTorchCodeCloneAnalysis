from train import gather_bn_weights
from copy import deepcopy

from test import *
from terminaltables import AsciiTable
from utils.prune_utils import *
# Prune settings
class opt():
    model_def = "config/yolov3-hand.cfg"
    data_config = "config/oxfordhand.data"
    pretrained_weights = 'checkpoints/yolov3_ckpt.pth'
    percent = 0.01
    save = "prunedRes"

data_config = parse_data_config(opt.data_config)
valid_path = data_config["valid"]
class_names = load_classes(data_config["names"])

### get the origin model's xx
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Darknet(opt.model_def).to(device)
if opt.pretrained_weights:
    if opt.pretrained_weights.endswith(".pth"):
        model.load_state_dict(torch.load(opt.pretrained_weights)['state_dict'])

eval_model = lambda model:evaluate(model, path=valid_path, iou_thres=0.5, conf_thres=0.01,
    nms_thres=0.5, img_size=model.img_size, batch_size=8)
obtain_num_parameters = lambda model:sum([param.nelement() for param in model.parameters()])

origin_model_metric = eval_model(model)
origin_nparameters = obtain_num_parameters(model)


if not os.path.exists(opt.save):
    os.makedirs(opt.save)
# 避免剪掉所有channel的最高阈值(每个BN层的gamma的最大值的最小值即为阈值上限)
modules = model.modules()
highest_thre = []
for index,module in enumerate(modules):
    if isinstance(module,nn.BatchNorm2d):
        if not isinstance(modules[index+1],channel_selection):
            highest_thre.append(module.weight.data.abs().max().item())

total,bn_weights = gather_bn_weights(model)
sorted_bn = torch.sort(bn_weights)[0]
highest_thre = min(highest_thre)
# 找到highest_thre对应的下标对应的百分比
percent_limit = (sorted_bn==highest_thre).nonzero().item()/len(bn_weights)
print(f'Threshold should be less than {highest_thre:.4f}.')
print(f'The corresponding prune ratio is {percent_limit:.3f}.')
thre_index = int(total * opt.percent)
thre = bn_weights[thre_index]

##get the threshold
pruned = 0
cfg = []
cfg_mask = []
for k, m in enumerate(model.modules()):
    if isinstance(m, nn.BatchNorm2d):
        weight_copy = m.weight.data.abs().clone()
        mask = weight_copy.gt(thre).float().cuda()
        pruned = pruned + mask.shape[0] - torch.sum(mask)
        m.weight.data.mul_(mask)
        m.bias.data.mul_(mask)
        cfg.append(int(torch.sum(mask)))
        cfg_mask.append(mask.clone())
        print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
            format(k, mask.shape[0], int(torch.sum(mask))))
print("Cfg:")
print(cfg)
pruned_ratio = pruned/total
##get the new model
index = 0
newmodeldef = deepcopy(model.module_defs)
for index,module_def in enumerate(newmodeldef):
    if module_def['type'] == 'bottleneck':
        block = int(module_def['block'])
        num = 3*block
        module_def['cfg'] = str(cfg[index, index+num])
        index +=num
assert index == len(cfg)
print('generated new module_def')
newmodel = Darknet(newmodeldef).to(device)
compact_nparameters = obtain_num_parameters(newmodel)

## generate the new model

old_modules = list(model.modules())
new_modules = list(newmodel.modules())
layer_id_in_cfg = 0
start_mask = torch.ones(3)
end_mask = cfg_mask[layer_id_in_cfg]
conv_count = 0

for layer_id in range(len(old_modules)):
    m0 = old_modules[layer_id]
    m1 = new_modules[layer_id]
    if isinstance(m0, nn.BatchNorm2d):
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        if idx1.size == 1:
            idx1 = np.resize(idx1,(1,))

        if isinstance(old_modules[layer_id + 1], channel_selection):
            # If the next layer is the channel selection layer, then the current batchnorm 2d layer won't be pruned.
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
            m1.running_mean = m0.running_mean.clone()
            m1.running_var = m0.running_var.clone()

            # We need to set the channel selection layer.
            m2 = new_modules[layer_id + 1]
            m2.indexes.data.zero_()
            m2.indexes.data[idx1.tolist()] = 1.0

            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):
                end_mask = cfg_mask[layer_id_in_cfg]
        else:
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()
            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                end_mask = cfg_mask[layer_id_in_cfg]
    elif isinstance(m0, nn.Conv2d):
        if conv_count == 0:
            m1.weight.data = m0.weight.data.clone()
            conv_count += 1
            continue
        if isinstance(old_modules[layer_id-1], channel_selection) or isinstance(old_modules[layer_id-1], nn.BatchNorm2d):
            # This convers the convolutions in the residual block.
            # The convolutions are either after the channel selection layer or after the batch normalization layer.
            conv_count += 1
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()

            # If the current convolution is not the last convolution in the residual block, then we can change the
            # number of output channels. Currently we use `conv_count` to detect whether it is such convolution.
            if conv_count % 3 != 1:
                w1 = w1[idx1.tolist(), :, :, :].clone()
            m1.weight.data = w1.clone()
            continue

        # We need to consider the case where there are downsampling convolutions.
        # For these convolutions, we just copy the weights.
        m1.weight.data = m0.weight.data.clone()
    elif isinstance(m0, nn.Linear):
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))

        m1.weight.data = m0.weight.data[:, idx0].clone()
        m1.bias.data = m0.bias.data.clone()

torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, os.path.join(opt.save, f'/prune_{opt.percent}_.pth.'))
#%%
# 在测试集上测试剪枝后的模型, 并统计模型的参数数量
compact_model_metric = eval_model(newmodel)
random_input = torch.rand((1, 3, model.img_size, model.img_size)).to(device)
model_forward_time, model_output = obtain_avg_forward_time(random_input, model)
newmodel_forward_time, compact_output = obtain_avg_forward_time(random_input, newmodel)
#%%
# 比较剪枝前后参数数量的变化、指标性能的变化
metric_table = [
    ["Metric", "Before", "After"],
    ["mAP", f'{origin_model_metric[2].mean():.6f}', f'{compact_model_metric[2].mean():.6f}'],
    ["Parameters", f"{origin_nparameters}", f"{compact_nparameters}"],
    ["Inference", f'{model_forward_time:.4f}', f'{newmodel_forward_time:.4f}']
]
print(AsciiTable(metric_table).table)
#%%
# 生成剪枝后的cfg文件并保存模型
pruned_cfg_name = opt.model_def.replace('/', f'/prune_{opt.percent}_')
pruned_cfg_file = write_cfg(pruned_cfg_name, [model.hyperparams.copy()] + newmodeldef)
print(f'Config file has been saved: {pruned_cfg_file}')

