from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap


def coco_eval(ref_json, sen_json):
    """
    计算coco评价指标

    :param ref_json:
    :param sen_json:
    :return:
    """
    coco = COCO(ref_json)
    coco_refs = coco.loadRes(sen_json)
    coco_eval_cap = COCOEvalCap(coco, coco_refs)
    coco_eval_cap.evaluate()
    return coco_eval_cap.eval.items()
