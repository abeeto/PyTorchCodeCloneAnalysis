from models import *
from utils.utils import *
import torch
import numpy as np
from copy import deepcopy
from test import evaluate
from terminaltables import AsciiTable
from DarknetModel import *
from utils.prune_utils import *
from prune import Prune_Model
class opt():
    model_def = "config/yolov3.cfg"
    data_config = "config/autodrive.data"
    pretrained_weights = 'checkpoints/darknet/checkpoint_55.pth'
    percent = 0.01
    save = "prunedRes"

if __name__ == '__main__':
    # %%
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(opt.model_def).to(device)
    model.load_state_dict(torch.load(opt.pretrained_weights)['state_dict'])

    data_config = parse_data_config(opt.data_config)
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    eval_model = lambda model: evaluate(model, path=valid_path, iou_thres=0.5, conf_thres=0.01,
                                        nms_thres=0.5, img_size=model.img_size, batch_size=6)
    obtain_num_parameters = lambda model: sum([param.nelement() for param in model.parameters()])

    origin_model_metric = eval_model(model)
    origin_nparameters = obtain_num_parameters(model)

    prune = Prune_Model(opt.model_def, opt.pretrained_weights, opt.percent)
    compact_model_defs, loose_model, compact_model = prune.getCompactModel()
    compact_nparameters = obtain_num_parameters(compact_model)
    eval_model(compact_model)

    random_input = torch.rand((1, 3, model.img_size, model.img_size)).to(device)
    pruned_forward_time, pruned_output = obtain_avg_forward_time(random_input, loose_model)
    compact_forward_time, compact_output = obtain_avg_forward_time(random_input, compact_model)

    diff = (pruned_output - compact_output).abs().gt(0.001).sum().item()
    if diff > 0:
        print('Something wrong with the pruned model!')

    # %%
    # 在测试集上测试剪枝后的模型, 并统计模型的参数数量
    compact_model_metric = eval_model(compact_model)

    # %%
    # 比较剪枝前后参数数量的变化、指标性能的变化
    metric_table = [
        ["Metric", "Before", "After"],
        ["mAP", f'{origin_model_metric[2].mean():.6f}', f'{compact_model_metric[2].mean():.6f}'],
        ["Parameters", f"{origin_nparameters}", f"{compact_nparameters}"],
        ["Inference", f'{pruned_forward_time:.4f}', f'{compact_forward_time:.4f}']
    ]
    print(AsciiTable(metric_table).table)

    # %%
    # 生成剪枝后的cfg文件并保存模型
    pruned_cfg_name = opt.model_def.replace('/', f'/prune_{opt.percent}_')
    pruned_cfg_file = write_cfg(pruned_cfg_name, [model.hyperparams.copy()] + compact_model_defs)
    print(f'Config file has been saved: {pruned_cfg_file}')

    compact_model_name = opt.pretrained_weights.replace('/', f'/prune_{opt.percent}_')
    torch.save(compact_model.state_dict(), compact_model_name)
    print(f'Compact model has been saved: {compact_model_name}')
