import torch
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from maskrcnn_benchmark.data.datasets.evaluation.coco import do_coco_evaluation

config_file = 'configs/da_faster_rcnn/e2e_da_faster_rcnn_R_50_C4_coco_to_MI3.yaml'
output_folder = 'da_cocoMI3'
box_only=False
iou_types=("bbox",)
predictions = torch.load('da_cocoMI3/inference/MI3_cocostyle/predictions.pth')
print(predictions)
cfg.merge_from_file(config_file)
expected_results=cfg.TEST.EXPECTED_RESULTS
expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL

data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=False)
for data_loader in data_loaders_val:
    dataset = data_loader.dataset           
    do_coco_evaluation(dataset,predictions,box_only,output_folder,iou_types,expected_results,expected_results_sigma_tol)
