import numpy as np
import torch
import argparse
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from config import cfg
from models.models_zoo import models
from data.build import make_dataloader
from utils.re_ranking import re_ranking
from utils.eval import extract_features, evaluate_core, cos_dist, euclid_dist


def evaluate(cfg, model, query_loader, test_loader):
    print("="*20, "Extracting features from query and gallery and evaluating" , "="*20)
    model.eval()
    with torch.no_grad():
        if cfg.RERANK.ENABLE:
            query_features, query_labels, query_camera_ids = extract_features(
                model, query_loader)
            gallery_features, gallery_labels, gallery_camera_ids = extract_features(
                model, test_loader)
            scores = re_ranking(probFea=query_features,
                                    galFea=gallery_features,
                                    k1=cfg.RERANK.K1,
                                    k2=cfg.RERANK.K2,
                                    lambda_value=cfg.RERANK.LAMBDA)
        else:
            query_features, query_labels, query_camera_ids = extract_features(model, query_loader)
            gallery_features, gallery_labels, gallery_camera_ids = extract_features(model, test_loader)
            if cfg.EVAL.DIST == 'euclid':
                scores = euclid_dist(query_features, gallery_features)
            elif cfg.EVAL.DIST == 'cos':
                scores = cos_dist(query_features, gallery_features)
            else:
                raise ValueError
    AP, CMC = 0, np.zeros(len(gallery_labels), dtype=np.float)
    ascending = False
    if cfg.RERANK.ENABLE or cfg.EVAL.DIST == 'euclid':
        ascending = True
    for i in range(len(query_labels)):
        ap, cmc = evaluate_core(scores[i], query_camera_ids[i], query_labels[i], gallery_camera_ids, gallery_labels, ascending=ascending)
        if cmc[0] == -1:
            continue
        CMC += cmc
        AP += ap
    CMC = CMC / len(query_labels)
    mAP = AP / len(query_labels)
    return CMC, mAP


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml', type=str, default='',
                        help='Dataset directory')
    args = parser.parse_args()
    if args.yaml:
        cfg.merge_from_file(args.yaml)
        cfg.freeze()
    print(cfg)
    train_loader, test_loader, query_loader = make_dataloader(cfg)
    model = models[cfg.MODEL.NAME](cfg.DATASET.NUM_CLASS, stride=cfg.MODEL.RESNET_STRIDE)
    model.load_state_dict(torch.load(cfg.TEST.CHECKPOINT))
    CMC, mAP = evaluate(cfg, model, query_loader, test_loader)
    print("Rank@1:{}, Rank@5: {}, Rank@10: {}, mAP: {}".format(CMC[0], CMC[4], CMC[9], mAP))