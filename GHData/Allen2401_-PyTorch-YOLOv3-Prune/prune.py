import torch
import numpy as np
from copy import deepcopy
from utils.parse_config import *
from test import *

class Prune_Model():

    def __init__(self,model_path, pth_path, percentage):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
       # load model params
        self.model = Darknet(model_path).to(device)
        if pth_path.endswith(".pth"):
            self.model.load_state_dict(torch.load(pth_path)['state_dict'])
        else:
            self.model.load_darknet_weights(pth_path)

        self.percent = percentage
        # get prune_list
        self.prune_list,self.CBL_list,self.conv_list = self.getPruneList()
        # sorted and get threshold
        self.threshold = self.getThreshold()

    def getPruneList(self):
        CBL_list = []
        conv_list = []
        ignore_list = []
        for module_i, module_def in enumerate(self.model.module_defs):
            if module_def["type"] == "convolutional":
                if module_def["batch_normalize"]:
                    CBL_list.append(module_i)
                else:
                    conv_list.append(module_i)
            elif module_def["type"] == "shortcut":
                ignore_list.append(module_i - 1)
                fromlayer = int(module_def["from"])
                if fromlayer < 0:
                    fromlayer += module_i
                if self.model.module_defs[fromlayer]["type"] == "convolutional":
                    ignore_list.append(fromlayer)

        prune_list = [idx for idx in CBL_list if idx not in ignore_list]
        return prune_list,CBL_list,conv_list

    def getThreshold(self):
        '''
        get all the batch normalization scales parameters
        :return:
        '''
        # quesition: how to get the parameters
        module_lists = self.model.module_list
        maxval_per_layer = []
        num = 0
        scales_size = [module_lists[idx][1].weight.data.shape[0] for idx in self.prune_list]
        scales_param = torch.zeros(sum(scales_size))
        # print(scales_size)
        for index, size in zip(self.prune_list, scales_size):
            params = module_lists[index][1].weight.data.abs().clone()
            scales_param[num:(num + size)] = params
            maxval_per_layer.append(params.max().cpu().numpy().tolist())
            num = num + size

        sorted_param, sort_index = torch.sort(scales_param)
        # the min of all max
        min_max = min(maxval_per_layer)
        pruning_length = int(len(sorted_param) * self.percent)
        threshold = sorted_param[pruning_length]
        # it presents that the whole layer will be pruning
        while threshold > min_max:
            self.percent = self.percent - 0.03
            print(f"Notice! To avoid the possibility of pruning whole layer,the percent will be decreased to %f",
                  self.percent)
            pruning_length = int(len(sorted_param) * self.percent)
            threshold = sorted_param[pruning_length]
        return threshold

    def generate_compact_defs(self):
        compact_model_defs = deepcopy(parse_model_config(self.model_path))
        head = compact_model_defs.pop(0)
        for module_i, (module_def, module) in enumerate(zip(self.model.module_defs, self.model.module_list)):

            if module_i in self.prune_list:
                params = module[1].weight.data.abs().item()
                remain_num = (params > self.threshold).sum()
                module_def["filters"] = remain_num

        compact_model_defs.insert(0, head)
        return compact_model_defs

    def getOutput_mask(self, index):
        '''
        默认index层是卷积层
        :param index:
        :return:
        '''
        module_list = self.model.module_list
        if index in self.prune_list:
            params = module_list[index][1].weight.data.abs()
            prune_mask = params > self.threshold
            return prune_mask
        else:
            prune_mask = torch.ones(module_list[index][0].weight.data.shape[0])
            return prune_mask

    def getInput_mask(self, index):

        if index == 0:
            return torch.ones(3)
        if self.model.module_defs[index - 1]['type'] == 'convolutional':
            return self.getOutput_mask(index - 1)
        elif self.model.module_defs[index - 1]['type'] == 'shortcut':
            return self.getOutput_mask(index - 2)
        elif self.model.module_defs[index - 1]['type'] == 'route':
            layers = [int(x) for x in self.model.module_defs[index - 1]["layers"].split(",")]
            mask = []
            for i,layer in enumerate(layers):
                if layer < 0:
                    layer += index -1
                    layers[i] = layer
                if self.model.module_defs[layer]['type'] == 'upsample' or self.model.module_defs[layer]['type'] == 'shortcut':
                    mask.append(self.getOutput_mask(layer - 1).tolist())
                else:
                    mask.append(self.getOutput_mask(layer).tolist())
            if len(layers) == 1:
                return self.getOutput_mask(layers[0])
            else:
                return torch.from_numpy(np.concatenate(np.array(mask)))

        else:
            print("error" + str(self.model.module_defs[index-1]['type'])+"   layer"+str(index))

    def getLooseModel(self):
        loose_model = deepcopy(self.model)
        route_idx = []
        save_activation = {}
        temp = 0
        total = 0
        prune = 0
        for index, module_def in enumerate(loose_model.module_defs):
            if module_def['type'] == 'route':
                layers = [int(x) for x in module_def["layers"].split(",")]
                if len(layers) == 1:
                    route_idx.append(index)

        for index, (module_def, module) in enumerate(zip(loose_model.module_defs, loose_model.module_list)):
            flag = 0
            if index in self.prune_list:
                flag = 1
                mask = self.getOutput_mask(index)
                bn_layer = module[1]
                bn_layer.weight.data.mul_(mask)
                save_activation[index] = F.leaky_relu(~mask * bn_layer.bias.data.clone(), 0.1)
                bn_layer.bias.data.mul(mask)
                temp = index
                total += len(mask)
                prune += mask.sum()
                print("index_layer:" + str(index) + "         before prune:" + str(len(mask)) + "          after prune:" + str(mask.sum()))

            elif index in route_idx:
                flag = 1
                # 需要把其上一层的信息路由下来
                temp = [int(x) for x in module_def["layers"].split(",")][0]
                if temp < 0:
                    temp += index
            if flag:
                next_model_def = loose_model.module_defs[index+1]
                next_module = loose_model.module_list[index+1]
                activation = save_activation[temp]
                if next_model_def['type'] == 'convolutional':
                    offset = next_module[0].weight.data.sum((2, 3)).matmul(activation.reshape(-1, 1)).reshape(-1)
                    if next_model_def['batch_normalize']:
                        next_module[1].running_mean.data.sub_(offset)
                    else:
                        next_module[0].bias.data.add_(offset)
        print("The remain percentage is " + str(prune/(total*1.0)))
        return loose_model

    def copyParams(self,loose_model,compact_model):
        '''
        该函数的作用是copy compact model的参数
        思考能不能将那个也添加进来
        :return:
        '''
        for index in self.CBL_list:

            output_idx = np.argwhere(self.getOutput_mask(index).numpy())[:, 0].tolist()
            input_idx = np.argwhere(self.getInput_mask(index).numpy())[:, 0].tolist()
            print("index layer" + str(index) + "    input_channel:" + str(len(input_idx)) + "    output_channel:" + str(len(output_idx)))
            bn_module = loose_model.module_list[index][1]
            bn_compact_module = compact_model.module_list[index][1]
            bn_compact_module.weight.data = bn_module.weight.data[output_idx].clone()
            bn_compact_module.bias.data = bn_module.bias.data[output_idx].clone()
            bn_compact_module.running_mean.data = bn_module.running_mean.data[output_idx].clone()
            bn_compact_module.running_var.data = bn_module.running_var.data[output_idx].clone()

            conv_module = loose_model.module_list[index][0]
            conv_compact_module = compact_model.module_list[index][0]
            temp = conv_module.weight.data[:, input_idx, :, :].clone()
            conv_compact_module.weight.data = temp[output_idx, :, :, :].clone()

        for index in self.conv_list:
            output_idx = np.argwhere(self.getOutput_mask(index).numpy())[:, 0].tolist()
            input_idx = np.argwhere(self.getInput_mask(index).numpy())[:, 0].tolist()
            print("index layer" + str(index) + "    input_channel:" + str(len(input_idx)) + "    output_channel:" + str(\
                len(output_idx)))
            conv_module = loose_model.module_list[index][0]
            conv_compact_module = compact_model.module_list[index][0]
            temp = conv_module.weight.data[:, input_idx, :, :].clone()
            conv_compact_module.weight.data = temp[output_idx, :, :, :].clone()
            if index not in self.CBL_list:
                conv_compact_module.bias.data = conv_module.bias.data[output_idx].clone()

        return compact_model

    def getCompactModel(self):
        # generate the compact defs
        compact_model_defs = self.generate_compact_defs()
        # generate compact model
        compact_model = Darknet(compact_model_defs)
        # get the loose model
        loose_model = self.getLooseModel()
        # copy  parameters
        self.copyParams(loose_model, compact_model)
        return compact_model_defs,loose_model,compact_model



