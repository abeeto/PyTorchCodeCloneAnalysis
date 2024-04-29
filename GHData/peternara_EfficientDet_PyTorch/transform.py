from models.efficientdet import EfficientDetBiFPN
#from models.efficientdet import EfficientDet
import torch
import numpy as np

import onnxruntime
from models.box_utils_numpy import hard_nms

def split_model_dict(model):
    backbone_dict = {}
    neck_dict = {}
    bbox_head_dict = {}

    num_backbone, num_neck, num_bbox_head = 0, 0, 0
    num_model = 0
    for k, v in model.items():
        if("backbone" in k):
            backbone_dict[k] = v
            num_backbone = num_backbone + 1
        elif('neck' in k):
            neck_dict[k] = v
            num_neck = num_neck + 1
        elif('bbox_head' in k):
            bbox_head_dict[k] = v

            num_bbox_head = num_bbox_head + 1
        num_model = num_model + 1
    assert num_model == num_backbone + num_neck + num_bbox_head
    return backbone_dict, neck_dict, bbox_head_dict

# combine model
def efficientdet_torch_to_onnx(input, onnx_path):
    det_bifpn = EfficientDetBiFPN(num_classes = 20, D_bifpn=2, W_bifpn=64)
    det_bifpn.backbone.set_swish(memory_efficient=False)

    efficient_data = torch.load('checkpoint_48.pth', map_location='cpu')
    model = efficient_data['state_dict']

    det_bifpn.load_state_dict(model)
    torch.onnx.export(det_bifpn, input, onnx_path, export_params=True,
                      keep_initializers_as_inputs=True,  opset_version=11, verbose=False)


def tensor_to_numpy(input_tensor):
    return input_tensor.cpu().numpy()


def Test_Onnx_Efficient_Det(dummy_input,
                            onnx_path,
                            iou_threshold = 0.5):
    ort_session = onnxruntime.InferenceSession(path_or_bytes = onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: tensor_to_numpy(dummy_input)}
    scores, classifications, anchors = ort_session.run(None, ort_inputs)

    
    boxes_scores = np.concatenate((anchors[0, :, :], scores[0, :, :]), axis=1)
    picked, box_picked = hard_nms(boxes_scores,
                          iou_threshold= iou_threshold)

    classify = np.amax(classifications[0, picked, :], axis=1)
    boxes = box_picked[:, :-1]
    score = box_picked[:, -1]

    print(classify.shape)
    print(boxes.shape)
    print(score.shape)

if __name__ == '__main__':
    dummy_input = torch.randn(1, 3, 512, 512)
    onnx_path = "efficientdet_d0.onnx"
    efficientdet_torch_to_onnx(dummy_input, onnx_path)
    print("Success transform to onnx")

    #Test_Onnx_Efficient_Det(dummy_input, onnx_path)