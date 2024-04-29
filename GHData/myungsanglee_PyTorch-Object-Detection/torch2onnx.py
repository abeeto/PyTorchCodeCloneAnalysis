import torch

from models.detector.yolov2 import YoloV2
from models.detector.yolov3 import YoloV3
from models.detector.yolov4_tiny import YoloV4TinyV4
from module.yolov2_detector import YoloV2Detector
from module.yolov3_detector import YoloV3Detector

from utils.yaml_helper import get_configs




if __name__ == '__main__':
    '''
    Convert to onnx
    '''
    # cfg = get_configs('configs/yolov4-tiny_focus-front.yaml')
    # ckpt = 'saved/yolov4-tiny_focus-front/version_0/checkpoints/epoch=99-step=81199.ckpt'
    # file_path = 'yolov4-tiny_focus-front_416.onnx'
    
    # model = YoloV4TinyV4(
    #     num_classes=cfg['num_classes'],
    #     num_anchors=len(cfg['anchors'])
    # )
    
    # model_module = YoloV3Detector.load_from_checkpoint(
    #     checkpoint_path=ckpt,
    #     model=model,
    #     cfg=cfg
    # )
    # model_module.eval()
    
    # input_names = ["input_1"]
    # input_sample = torch.randn((1, 3, 416, 416), device='cpu')
    # model_module.to_onnx(file_path, input_sample, export_params=True, 
    #                      opset_version=12, verbose=False, input_names=input_names, 
    #                      do_constant_folding=False)
    
    
    import onnx
    path = 'yolov4-tiny_focus-front_416.onnx'
    model = onnx.load(path)
    print(model.ir_version)
    