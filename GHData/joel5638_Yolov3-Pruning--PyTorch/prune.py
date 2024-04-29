# coding: utf-8
import argparse

from models import *
from utils.datasets import *
from utils.utils import *
from utils.parse_config import *

""" Slim Principle
(1) Use global threshold to control pruning ratio
(2) Use local threshold to keep at least 10% unpruned 
"""


def list_without_bracket(temp_line):
    if isinstance(temp_line, np.ndarray):
        temp_line = temp_line.ravel('F') # flatten
        temp_line = temp_line.tolist() # np.array to list

    for i in range(len(temp_line)):
        line = str(temp_line).replace(']','').replace('[','') # Remove bracket from list which became list
        if i != len(temp_line)-1:
            line += ','

    return line


def route_conv(layer_index, module_defs):
    """ find the convolutional layers connected by route layer
    When route layer's attributes has only one, it outputs FM of layer indexed by the value.
    When its attribute has more than two, it returns concatenated FM of layers indexed by the value.
    """

    module_def = module_defs[layer_index]
    mtype = module_def['type']   

    before_conv_id = []
    if mtype in ['convolutional', 'shortcut', 'upsample', 'maxpool']:
        if module_defs[layer_index-1]['type'] == 'convolutional':
            return [layer_index-1]
        before_conv_id += route_conv(layer_index-1, module_defs)

    elif mtype == "route":
        # put 'layers' attribute to layer_is
        layer_is = [int(x)+layer_index if int(x) < 0 else int(x) for x in module_defs[layer_index]['layers']]
        print('layer_is: ', layer_is)
        for layer_i in layer_is:
            if module_defs[layer_i]['type'] == 'convolutional':
                before_conv_id += [layer_i] # append layer_i to before_conv_id
            else:
                before_conv_id += route_conv(layer_i, module_defs)
    print('before_conv_id: ', before_conv_id)

    return before_conv_id


def write_model_cfg(old_path, new_path, new_module_defs):
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    print('Writing model configuration file...')
    lines = []
    with open(old_path, 'r') as fp:
        old_lines = fp.readlines()
    for _line in old_lines:
        if "[convolutional]" in _line:
            break
        lines.append(_line)

    for i, module_def in enumerate(new_module_defs):
        
        mtype = module_def['type']
        lines.append("[{}]\n".format(mtype))
        #print("layer:", i, mtype)
        if mtype == "convolutional":
            bn = 0
            filters = module_def['filters']
            bn = int(module_def['batch_normalize'])
            if bn:
                lines.append("batch_normalize={}\n".format(bn))
                filters = torch.sum(module_def['mask']).cpu().numpy().astype('int')        
            lines.append("filters={}\n".format(filters))
            lines.append("size={}\n".format(module_def['size']))
            lines.append("stride={}\n".format(module_def['stride']))
            lines.append("pad={}\n".format(module_def['pad']))
            lines.append("activation={}\n\n".format(module_def['activation']))
        elif mtype == "shortcut":
            _temp_from = list_without_bracket(module_def['from'])
            lines.append("from={}\n".format(_temp_from))
            lines.append("activation={}\n\n".format(module_def['activation']))   
        elif mtype == 'route':
            _temp_layers = list_without_bracket(module_def['layers'])
            lines.append('layers={}\n\n'.format(_temp_layers))

        elif mtype == 'upsample':
            lines.append("stride={}\n\n".format(module_def['stride']))
        elif mtype == 'maxpool':
            lines.append("stride={}\n".format(module_def['stride']))
            lines.append("size={}\n\n".format(module_def['size']))
        elif mtype == 'yolo':
            _temp_mask = list_without_bracket(module_def['mask'])
            lines.append("mask = {}\n".format(_temp_mask))
            _temp_anchors = list_without_bracket(module_def['anchors'])
            lines.append("anchors = {}\n".format(_temp_anchors))
            lines.append("classes = {}\n".format(module_def['classes']))
            lines.append("num = {}\n".format(module_def['num']))
            lines.append("jitter = {}\n".format(module_def['jitter']))
            lines.append("ignore_thresh = {}\n".format(module_def['ignore_thresh']))
            lines.append("truth_thresh = {}\n".format(module_def['truth_thresh']))
            lines.append("random = {}\n\n".format(module_def['random']))

    with open(new_path, "w") as f:
        f.writelines(lines)


def test(
        cfg,
        weights=None,
        img_size=416,
        save=None,
        overall_ratio=0.5,
        perlayer_ratio=0.1):

    """prune yolo and generate cfg, weights
    """
    # Create directory if not existed
    if save is not None:
        if not os.path.exists(save):
            os.makedirs(save)

    device = torch_utils.select_device()  # --> cuda?

    # Initialize model
    model = Darknet(cfg, img_size).to(device) # Hand over configuration file and image size

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        _state_dict = torch.load(weights, map_location=device)['model']
        model.load_state_dict(_state_dict)
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

    # output a new cfg file
    total = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):  # Total, 72 BatchNorm2d layers
            total += m.weight.data.shape[0]  # channels numbers
    bn = torch.zeros(total)  # 총 BN layer의 channel 갯수만큼 tensor with zero value
    index = 0

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]  # m.weight.data.shape=  torch.Size([32])
            # print(m.weight.data.shape) # tensor list which has shape[0] number of values
            bn[index:(index+size)] = m.weight.data.abs().clone() #
            # input("Why it's not torch.Size[3,32] but just [32]?" - RGB is 3 channel)
            index += size

    sorted_bn, sorted_index = torch.sort(bn)
    thresh_index = int(total*overall_ratio) # total num of all BN layers * overall_ratio
    thresh = sorted_bn[thresh_index].cuda()

    print("--"*30)
    print()

    # Pruning channels in each layer
    # Record mask for chosen channel in each BN layer of proned_module_defs[i]["mask"]
    proned_module_defs = model.module_defs
    for i, (module_def, module) in enumerate(zip(model.module_defs, model.module_list)):
        mtype = module_def['type']
        if mtype == 'convolutional':
            bn = int(module_def['batch_normalize'])
            if bn:  # if 'batch_normalize: 1' --> What is the meaning of 1: there is BN after Conv layer.
                m = getattr(module, 'BatchNorm2d')
                weight_copy = m.weight.data.abs().clone()  # copy torch.size([channel num])
                channels = weight_copy.shape[0]  # num of channels of one BN layer
                min_channel_num = int(channels * perlayer_ratio) if int(channels * perlayer_ratio) > 0 else 1 # usually perlayer_ratio: 0.1
                mask = weight_copy.gt(thresh).float().cuda()  # masking for chosen (important) channels whose value is over than threshold

                # print('channels: {} \t perlayer_ratio: {} \t min_channel_num: {} '.format(channels, perlayer_ratio, min_channel_num))
                if int(torch.sum(mask)) < min_channel_num:  # if num of will-be-pruned channels is smaller than minimum pruned-channels
                    _, sorted_index_weights = torch.sort(weight_copy, descending=True)  # From biggest number to lowest
                    # print('sorted_index_weights: ', sorted_index_weights)
                    # print('sorted_index_weights[:min_channel_num]: ', sorted_index_weights[:min_channel_num]) # 최소 있어야 할 채널 수
                    # print(sorted_index_weights)
                    mask[sorted_index_weights[:min_channel_num]] = 1. # Index of important channels --> mask = 1

                proned_module_defs[i]['mask'] = mask.clone()  # 특정 layer i 에서 'mask' 값 복사사


        elif mtype == 'shortcut': # skipped layer
            layer_i = int(str(module_def['from'][0])) + i
            # print('i: {} \t layer_i: {} \t mtype: {}'.format(i, layer_i, mtype))
            proned_module_defs[i]['is_access'] = False
    

    # Check shortcut(skip connection) related layers' mask and sum, then renew those layers' mask
    layer_number = len(proned_module_defs)  # 125
    for i in range(layer_number-1, -1, -1): # descending order
        mtype = proned_module_defs[i]['type']
        # print('i:', i, ' mtype: ', mtype)
        if mtype == 'shortcut':
            if proned_module_defs[i]['is_access']: 
                continue

            Merge_masks = []  # reset in every layer
            layer_i = i

            # Check all shortcut related layers' mask and sum, then set 1 which has value > 0
            while mtype == 'shortcut':  # ['is_access'] = False  --> i:
                proned_module_defs[layer_i]['is_access'] = True  # checked

                if proned_module_defs[layer_i-1]['type'] == 'convolutional':  # layer-1 = 73, 60, 35, 11, 4
                    bn = int(proned_module_defs[layer_i-1]['batch_normalize'])
                    if bn:
                        Merge_masks.append(proned_module_defs[layer_i-1]["mask"].unsqueeze(0))  # [] --> [ [] ]
                        

                layer_i = int(str(proned_module_defs[layer_i]['from'][0])) + layer_i # 62
                mtype = proned_module_defs[layer_i]['type']

                if mtype == 'convolutional':
                    bn = int(proned_module_defs[layer_i]['batch_normalize'])
                    if bn:
                        Merge_masks.append(proned_module_defs[layer_i]["mask"].unsqueeze(0))
                        

            if len(Merge_masks) > 1:
                Merge_masks = torch.cat(Merge_masks, 0)  # each tensor combined to one tensor but in different list.                
                merge_mask = (torch.sum(Merge_masks, dim=0) > 0).float().cuda() # If torch.sum - element is over 1 -> True --> 1                
            else:
                merge_mask = Merge_masks[0].float().cuda()                

            layer_i = i  # First shortcut layer's i-th
            mtype = 'shortcut'

            # shortcut connected layer's mask equals to merge_mask (all same) / 5, 9, 3, 2
            while mtype == 'shortcut':
                if proned_module_defs[layer_i-1]['type'] == 'convolutional':
                    bn = int(proned_module_defs[layer_i-1]['batch_normalize'])
                    if bn:
                        proned_module_defs[layer_i-1]["mask"] = merge_mask
                        print('layer_i {} \t merge_mask.shape: {} '.format(layer_i, merge_mask.shape))

                layer_i = int(str(proned_module_defs[layer_i]['from'][0]))+layer_i
                mtype = proned_module_defs[layer_i]['type']
                print('updated layer_i: ', layer_i)

                if mtype == 'convolutional': 
                    bn = int(proned_module_defs[layer_i]['batch_normalize'])
                    if bn:     
                        proned_module_defs[layer_i]["mask"] = merge_mask
                        print('layer_i {} \t mtype: {} \t merge_mask.shape: {} '.format(layer_i, mtype,merge_mask.shape))
    # Put 'mask_before' attributes to convolutional layer considering route layer
    # Update number of output FM channel from previous layer
    print('\nRoute_conv Start!')
    for i, (module_def, module) in enumerate(zip(model.module_defs, model.module_list)):
        mtype = module_def['type']
        if mtype == 'convolutional':
            bn = int(module_def['batch_normalize'])
            if bn:
                proned_module_defs[i]['mask_before'] = None
                mask_before = []
                if i > 0:
                    conv_indexs = route_conv(i, proned_module_defs)
                    for conv_index in conv_indexs:
                        mask_before += proned_module_defs[conv_index]["mask"].clone().cpu().numpy().tolist()                        
                    proned_module_defs[i]['mask_before'] = torch.tensor(mask_before).float().cuda()  # input proned's mask


    # Save new configuration file about pruned model
    output_cfg_path = os.path.join(save, "prune.cfg")
    write_model_cfg(cfg, output_cfg_path, proned_module_defs)

    # Load new cfg file of pruned model
    pruned_model = Darknet(output_cfg_path, img_size).to(device)

    # Copy masked before-pruned weights to after-pruned weights
    for i, (module_def, old_module, new_module) in enumerate(zip(proned_module_defs, model.module_list, pruned_model.module_list)):  
        mtype = module_def['type']

        if mtype == 'convolutional':
            bn = int(module_def['batch_normalize'])
            if bn:
                new_norm = getattr(new_module, 'BatchNorm2d')
                old_norm = getattr(old_module, 'BatchNorm2d')

                new_conv = getattr(new_module, 'Conv2d')
                old_conv = getattr(old_module, 'Conv2d')

                idx1 = np.squeeze(np.argwhere(np.asarray(module_def['mask'].cpu().numpy())))
                if i > 0:
                    idx2 = np.squeeze(np.argwhere(np.asarray(module_def['mask_before'].cpu().numpy())))
                    new_conv.weight.data = old_conv.weight.data[idx1.tolist()][:, idx2.tolist(), :, :].clone()
                    print(i, '\t new: ', new_conv.weight.data.shape, '\t old: ', old_conv.weight.data.shape)


                else:
                    new_conv.weight.data = old_conv.weight.data[idx1.tolist()].clone()
                    print(i, '\t new: ', new_conv.weight.data.shape, '\t old: ', old_conv.weight.data.shape)

                new_norm.weight.data = old_norm.weight.data[idx1.tolist()].clone()
                new_norm.bias.data = old_norm.bias.data[idx1.tolist()].clone()
                new_norm.running_mean = old_norm.running_mean[idx1.tolist()].clone()
                new_norm.running_var = old_norm.running_var[idx1.tolist()].clone()

            else:  # No BN after Conv2d
                new_conv = getattr(new_module, 'Conv2d')
                old_conv = getattr(old_module, 'Conv2d')
                idx2 = np.squeeze(np.argwhere(np.asarray(proned_module_defs[i-1]['mask'].cpu().numpy())))
                new_conv.weight.data = old_conv.weight.data[:,idx2.tolist(),:,:].clone()
                new_conv.bias.data = old_conv.bias.data.clone()
                print(i, '\t new: ', new_conv.weight.data.shape, '\t old: ', old_conv.weight.data.shape)
                print('layer index: ', i, "entire copy")

    print('--'*30)
    print('prune done!')    
    print('pruned ratio %.3f' % overall_ratio)
    prune_weights_path = os.path.join(save, "prune.pt")    
    _pruned_state_dict = pruned_model.state_dict()
    torch.save(_pruned_state_dict, prune_weights_path)
    print("Done!") 

    # test the pruned model
    pruned_model.eval()
    img_path = "../DL-DATASET/etri-safety_system/distort/images/640x480/GH020005_006.jpg"
    
    org_img = cv2.imread(img_path)  # BGR
    img, _, _ = letterbox(org_img, [img_size,img_size])

    # Normalize
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    imgs = torch.from_numpy(img).unsqueeze(0).to(device)
    _, _, height, width = imgs.shape  # batch size, channels, height, width

    # Run model
    inf_out, train_out = pruned_model(imgs)  # inference and training outputs
    # Run NMS
    output = non_max_suppression(inf_out, conf_thres=0.3, iou_thres=0.5)
    # Statistics per image
    for si, pred in enumerate(output):
        if pred is None:
            continue
        if True:
            box = pred[:, :4].clone()  # xyxy
            scale_coords(imgs[si].shape[1:], box, org_img.shape[:2])  # to original shape

            for di, d in enumerate(pred):
                print('di: {} \t d:{} '.format(di, d) )

                category_id = int(d[5])
                left, top, right, bot = [float(x) for x in box[di]]
                confidence = float(d[4])

                cv2.rectangle(org_img, (int(left), int(top)), (int(right), int(bot)),
                                (255, 0, 0), 2)
                cv2.putText(org_img, str(category_id) + ":" + str('%.1f' % (float(confidence) * 100)) + "%", (int(left), int(top) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)  

        # cv2.imshow("result", org_img)
        # cv2.waitKey(0)
        cv2.imwrite('result_{}'.format(img_path), org_img)

    # Convert .pt to .weights:
    print('Saving weight....')
    prune_c_weights_path = os.path.join(save, "prune.weights")
    save_weights(pruned_model, prune_c_weights_path)
    print('Done!')
    

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parser = argparse.ArgumentParser(description='PyTorch Slimming Yolov3 prune')
    parser.add_argument('--cfg', type=str, default='VisDrone2019/yolov3-spp3.cfg', help='cfg file path')
    parser.add_argument('--weights', type=str, default='yolov3-spp3_final.weights', help='path to weights file')
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--save', default='prune', type=str, metavar='PATH', help='path to save pruned model (default: none)')
    parser.add_argument('--overall_ratio', type=float, default=0.5, help='scale sparse rate (default: 0.5)')    
    parser.add_argument('--perlayer_ratio', type=float, default=0.1, help='minimal scale sparse rate (default: 0.1) to prevent disconnect')    
    
    opt = parser.parse_args()
    opt.save += "_{}_{}".format(opt.overall_ratio, opt.perlayer_ratio)

    print(opt)
    file_opt = "opts.txt"

    save_opts(opt.save, file_opt, opt)

    with torch.no_grad():
        test(
            opt.cfg,
            opt.weights,
            opt.img_size,
            opt.save,
            opt.overall_ratio,
            opt.perlayer_ratio,
        )
