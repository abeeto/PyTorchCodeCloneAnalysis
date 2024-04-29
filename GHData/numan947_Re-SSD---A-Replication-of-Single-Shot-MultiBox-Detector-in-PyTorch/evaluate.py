from commons import *
import torch
import os
from pprint import PrettyPrinter
from data import PascalDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from output_processing import perform_nms
from train import clear_cuda
from tabulate import tabulate


def calculate_11_point_AP(cumulative_precision, cumulative_recall):
    device = torch.device('cpu')
    recall_thresholds = torch.arange(start=0, end=1.1, step=0.1).tolist()
    precisions = torch.zeros(len(recall_thresholds), dtype=torch.float).to(device)
    
    for i, t in enumerate(recall_thresholds):
        recalls_above_t = cumulative_recall>=t
        if recalls_above_t.any():
            precisions[i] = cumulative_precision[recalls_above_t].max()
        else:
            precisions[i] = 0
    return precisions.mean()

def calculate_all_point_AP(cumulative_precision, cumulative_recall):
    device = torch.device('cpu')
    ap = 0.0
    cr_level = 0
    k = 0
    while(True):
        recalls_above_t = cumulative_recall>cr_level
        # print(recalls_above_t)
        if not recalls_above_t.any():
            break
        valid_cum_prec = cumulative_precision[recalls_above_t]
        value, ind = torch.max(valid_cum_prec, dim=0)
        
        valid_cum_recall = cumulative_recall[recalls_above_t][valid_cum_prec == value]
        
        ap+=value.item()*(valid_cum_recall.max() - cr_level)
        cr_level = valid_cum_recall.max()
        
        k+=1
        if k>1000000:
            """This should never happen. But again if it's happening then, either 
            my implementation is wrong or your dataset have a class which have more than
            1000000 instances -- which shouldn't happen for fairly small datasets"""
            print("LOOPING PROBLEM IN CALCULATE ALL POINT AP")
            break
        
    return ap

def calculate_mAP(detected_boxes, detected_labels, detected_scores, true_boxes, true_labels, true_diffs, mAP = "11pt"):
    device = torch.device('cpu')
    assert len(detected_boxes) == len(detected_labels) ==len(detected_scores) == len(true_boxes) == len(true_diffs) == len(true_diffs)

    mAP = mAP.upper()
    assert mAP in {"11PT", "ALLPT"}
    n_classes = len(label_map)


    true_images = list()
    for i in range(len(true_labels)):
        true_images.extend([i]*true_labels[i].size(0))
    true_images = torch.LongTensor(true_images).to(device)
    true_boxes = torch.cat(true_boxes, dim=0)
    true_labels = torch.cat(true_labels, dim=0)
    true_diffs = torch.cat(true_diffs, dim=0)
    
    assert true_images.size(0) == true_labels.size(0) == true_boxes.size(0) == true_diffs.size(0)
    
    
    detected_images = list()
    for i in range(len(detected_labels)):
        detected_images.extend([i]*detected_labels[i].size(0))
    detected_images = torch.LongTensor(detected_images).to(device)
    detected_boxes = torch.cat(detected_boxes, dim=0)
    detected_labels = torch.cat(detected_labels, dim=0)
    detected_scores = torch.cat(detected_scores, dim=0)
    
    assert detected_images.size(0) == detected_boxes.size(0) == detected_labels.size(0) == detected_scores.size(0)
    
    print("\nAverage Precision Calculation:....preprocessing completed!")
    print("Initiating AP calculation ....")
    
    average_precisions = torch.zeros((n_classes-1), dtype=torch.float)
    
    for c in tqdm(range(1, n_classes)):
        true_class_images = true_images[true_labels == c]
        true_class_boxes = true_boxes[true_labels == c]
        true_class_diffs = true_diffs[true_labels == c]
        n_easy_class_objects = (~true_class_diffs).sum().item()
        
        detected_class_images = detected_images[detected_labels == c]
        detected_class_boxes = detected_boxes[detected_labels == c]
        detected_class_scores = detected_scores[detected_labels == c]
        n_class_detection = detected_class_boxes.size(0)
        
        if n_class_detection == 0:
            continue
        
        detected_class_scores, sort_ind = detected_class_scores.sort(dim=0, descending=True)
        detected_class_images = detected_class_images[sort_ind]
        detected_class_boxes = detected_class_boxes[sort_ind]
        
        
        true_class_boxes_detected = torch.zeros((true_class_boxes.size(0)), dtype=torch.bool).to(device)
        true_positives = torch.zeros((n_class_detection), dtype=torch.float).to(device)
        false_positives = torch.zeros((n_class_detection), dtype=torch.float).to(device)
    
        for d in range(n_class_detection):
            current_box = detected_class_boxes[d].unsqueeze(0)
            current_image = detected_class_images[d]
            
            object_boxes = true_class_boxes[true_class_images == current_image]
            object_diffs = true_class_diffs[true_class_images == current_image]
            
            if object_boxes.size(0) == 0:
                false_positives[d] = 1
                continue
            
            overlaps = find_jaccard_overlap(current_box, object_boxes)
            max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)
            
            original_ind = torch.LongTensor(range(true_class_boxes.size(0)))[true_class_images == current_image][ind]
            
            if max_overlap.item() > 0.5:
                if object_diffs[ind] == 0:
                    if true_class_boxes_detected[original_ind] == 0:
                        true_positives[d] = 1
                        true_class_boxes_detected[original_ind] = 1
                    else:
                        false_positives[d] = 1
            else:
                false_positives[d] = 1 
            
        
        cumul_true_positives = torch.cumsum(true_positives, dim=0)
        cumul_false_positives = torch.cumsum(false_positives, dim=0)
        
        cum_precision = cumul_true_positives/(cumul_true_positives+cumul_false_positives+1e-16)
        cum_recall = cumul_true_positives/n_easy_class_objects
        
        if mAP == "11PT":
            average_precisions[c-1] = calculate_11_point_AP(cum_precision, cum_recall)
        elif mAP == "ALLPT":
            average_precisions[c-1] = calculate_all_point_AP(cum_precision, cum_recall)

    mean_average_precision = average_precisions.mean().item()
    
    average_precisions = {rev_label_map[c+1]:v for c, v in enumerate(average_precisions.tolist())}
    
    
    return average_precisions, mean_average_precision


def test(model, mAP="11PT", resize_dims=(300,300)):
    model = model.to(device)
    model.eval()

    data_folder = "./output/"
    keep_difficult = True
    batch_size = 1
    workers = 4*torch.cuda.device_count()
    pin_memory = True

    test_dataset = PascalDataset(data_folder, split="test", keep_difficult=keep_difficult, resize_dims=resize_dims)
    
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=test_dataset.collate_fn,
                             num_workers=workers,pin_memory=pin_memory)



    detected_boxes = list()
    detected_labels = list()
    detected_scores = list()

    true_boxes = list()
    true_labels = list()
    true_diffs = list()


    with torch.no_grad():
        for i, (images, boxes,labels, diffs) in enumerate(tqdm(test_loader, total=len(test_loader))):
            try:
                images = images.to(device)
                predicted_locs, predicted_scores = model(images)
                detected_boxes_batch, detected_labels_batch, detected_scores_batch = perform_nms(model.priors_cxcy, model.n_classes, predicted_locs, predicted_scores, min_score=0.01, max_overlap=0.45, top_k=200)
            except(RuntimeError) as e:
                if 'out of memory' in str(e):
                    print("CUDA OUT OF MEMORY FOR image {}".format(test_dataset.images[i]))
                    clear_cuda()
                    continue
                else:
                    raise e
            clear_cuda()
            # boxes = [b.to(device) for b in boxes]
            # labels = [l.to(device) for l in labels]
            # diffs = [d.to(device) for d in diffs]


            detected_boxes.extend(detected_boxes_batch)
            detected_labels.extend(detected_labels_batch)
            detected_scores.extend(detected_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)
            true_diffs.extend(diffs)
            
        
        APs, mAP = calculate_mAP(detected_boxes, detected_labels, detected_scores, true_boxes, true_labels, true_diffs, mAP)

        pp = PrettyPrinter()
        pp.pprint(APs)
        print("\nMean Average Precision (mAP): {:.3f}".format(mAP))
        
        

if __name__ == "__main__":
    precisions = [1, 0.5, 0.6666, 0.5, 0.4,0.3333,0.2857,0.25,0.2222,0.3,0.2727,0.3333,0.3846,0.4285,0.4,0.375,0.3529,0.3333,0.3157,0.3,0.2857,0.2727,0.3043,0.2916]
    recalls =    [0.0666, 0.0666, 0.1333, 0.1333, 0.1333, 0.1333, 0.1333, 0.1333, 0.1333, 0.2, 0.2, 0.2666, 0.3333, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4666, 0.4666]
    
    print(len(precisions))
    print(len(recalls))
    
    precisions = torch.FloatTensor(precisions).to(device)
    recalls = torch.FloatTensor(recalls).to(device)
    
    print(calculate_all_point_AP(precisions, recalls))