from tracemalloc import start
import numpy as np
import cv2
from dataset import iou


colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
#use [blue green red] to represent different classes

def visualize_pred(windowname, pred_confidence, pred_box, ann_confidence, ann_box, image_, boxs_default):
    '''
    input:
    windowname      -- the name of the window to display the images
    pred_confidence -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    pred_box        -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    ann_confidence  -- the ground truth class labels, [num_of_boxes, num_of_classes]
    ann_box         -- the ground truth bounding boxes, [num_of_boxes, 4]
    image_          -- the input image to the network
    boxs_default    -- default bounding boxes, [num_of_boxes, 8]
    '''

    _, class_num = pred_confidence.shape
    #class_num = 4
    class_num = class_num-1
    #class_num = 3 now, because we do not need the last class (background)
    image_ = image_ * 255
    image = np.transpose(image_, (1,2,0)).astype(np.uint8)
    image1 = np.zeros(image.shape,np.uint8)
    image2 = np.zeros(image.shape,np.uint8)
    image3 = np.zeros(image.shape,np.uint8)
    image4 = np.zeros(image.shape,np.uint8)
    image1[:]=image[:]
    image2[:]=image[:]
    image3[:]=image[:]
    image4[:]=image[:]
    #image1: draw ground truth bounding boxes on image1
    #image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
    #image3: draw network-predicted bounding boxes on image3
    #image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
    bbox_8_gt = get_absolute_gtbox(ann_box,boxs_default)
    bbox_8_pred_gt = get_absolute_gtbox(pred_box,boxs_default)
    # Construct a box = [x1_center, y1_center, width, height, x1_min, y1_min, x1_max, y1_max]
    
    height, width, _ = image.shape
    # print('visual',image.shape)
    #draw ground truth
    for i in range(len(ann_confidence)):
        for j in range(class_num):
            if ann_confidence[i,j]>0.5: #if the network/ground_truth has high confidence on cell[i] with class[j]
                #image1: draw ground truth bounding boxes on image1
                #image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
                start_point = (int(bbox_8_gt[i,4]*width), int(bbox_8_gt[i,5]*height)) #top left corner, x1<x2, y1<y2
                end_point = (int(bbox_8_gt[i,6]*width), int(bbox_8_gt[i,7]*height)) #bottom right corner
                color = colors[j] #use red green blue to represent different classes
                thickness = 2
                image1 = cv2.rectangle(image1, start_point, end_point, color, thickness)

                start_point_2 = (int(boxs_default[i,4]*width), int(boxs_default[i,5]*height)) #top left corner, x1<x2, y1<y2
                end_point_2 = (int(boxs_default[i,6]*width), int(boxs_default[i,7]*height)) #bottom right corner
                image2 = cv2.rectangle(image2, start_point_2, end_point_2, color, thickness)
    
    #pred
    for i in range(len(pred_confidence)):
        for j in range(class_num):
            if pred_confidence[i,j]>0.5:
                #image3: draw network-predicted bounding boxes on image3
                #image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
                start_point_3 = (int(bbox_8_pred_gt[i,4]*width), int(bbox_8_pred_gt[i,5]*height)) #top left corner, x1<x2, y1<y2
                end_point_3 = (int(bbox_8_pred_gt[i,6]*width), int(bbox_8_pred_gt[i,7]*height)) #bottom right corner
                color2 = colors[j] #use red green blue to represent different classes
                thickness2 = 2
                image3 = cv2.rectangle(image3, start_point_3, end_point_3, color2, thickness2)

                start_point_4 = (int(boxs_default[i,4]*width), int(boxs_default[i,5]*height)) #top left corner, x1<x2, y1<y2
                end_point_4 = (int(boxs_default[i,6]*width), int(boxs_default[i,7]*height)) #bottom right corner
                image4 = cv2.rectangle(image4, start_point_4, end_point_4, color2, thickness2)
    
    #combine four images into one
    h,w,_ = image1.shape
    image = np.zeros([h*2,w*2,3], np.uint8)
    image[:h,:w] = image1
    image[:h,w:] = image2
    image[h:,:w] = image3
    image[h:,w:] = image4
    # cv2.imshow(windowname+" [[gt_box,gt_dft],[pd_box,pd_dft]]",image)
    # cv2.waitKey(1)
    cv2.imwrite(windowname+".jpg",image)

def get_absolute_gtbox(box_,boxs_default):
    '''
    input:
    box_                -- ground truth bbox in relative coordinates or predicted bbox of SSD 
    box_default         -- default SSD boxes

    output:
    bbox_8              -- ground truth bbox not respect to SSD default bbox
    '''
    dx = box_[:,0]
    dy = box_[:,1]
    dw = box_[:,2]
    dh = box_[:,3]

    px = boxs_default[:,0]
    py = boxs_default[:,1]
    pw = boxs_default[:,2]
    ph = boxs_default[:,3]

    gx = pw * dx + px
    gy = ph * dy + py
    gw = pw * np.exp(dw)
    gh = ph * np.exp(dh)

    # Construct a box = [x1_center, y1_center, width, height, x1_min, y1_min, x1_max, y1_max] to utilize IOU from dataset.py
    bbox_8 = np.zeros_like(boxs_default,dtype=np.float32)
    bbox_8[:,0] = gx
    bbox_8[:,1] = gy
    bbox_8[:,2] = gw
    bbox_8[:,3] = gh
    bbox_8[:,4] = gx - gw/2.0
    bbox_8[:,5] = gy - gh/2.0
    bbox_8[:,6] = gx + gw/2.0
    bbox_8[:,7] = gy + gh/2.0
    return bbox_8

def non_maximum_suppression(confidence_, box_, boxs_default, overlap=0.3, threshold=0.8):
    '''
    input:
    confidence_  -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    box_         -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    boxs_default -- default bounding boxes, [num_of_boxes, 8]
    overlap      -- if two bounding boxes in the same class have iou > overlap, then one of the boxes must be suppressed
    threshold    -- if one class in one cell has confidence > threshold, then consider this cell carrying a bounding box with this class.
    
    output:
    out_confidence -- the final confidence after NMS, [num_of_boxes, num_of_classes]
    out_bbox       -- the final bounding boxes from SSD after NMW, [num_of_boxes, 4]
    '''

    bbox_8 = get_absolute_gtbox(box_,boxs_default)
    
    out_bbox = np.zeros_like(box_,dtype=np.float32)
    out_confidence = np.zeros_like(confidence_,dtype=np.float32)
    out_confidence[:,-1] = 1 #the default class for all cells is set to "background"
    while True:
        highest = np.argmax(confidence_[:,0:-1])
        r, c = divmod(highest, confidence_[:,0:-1].shape[1])
        if confidence_[r,c] >= threshold:
            out_bbox[r,:] = box_[r,:]
            out_confidence[r,:] = confidence_[r,:]

            ious = iou(bbox_8,bbox_8[r,4],bbox_8[r,5],bbox_8[r,6],bbox_8[r,7])
            ious_bigger_threshold = np.where(ious > overlap)[0]
            box_[ious_bigger_threshold,:] = [0., 0., 0., 0.]
            confidence_[ious_bigger_threshold,:] = [0, 0, 0, 1]
            bbox_8[ious_bigger_threshold,:] = [0., 0., 0., 0., 0., 0., 0., 0.]
        else:
            return out_confidence, out_bbox

def write_txt(pred_box, boxs_default, pred_confidence, img_name, height_origin, width_origin,txt):
    '''
    input:
    pred_box         -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    boxs_default     -- default bounding boxes, [num_of_boxes, 8]
    pred_confidence  -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    img_name         -- image name to produce txt file name
    height_origin    -- the original height of the image
    width_origin     -- the original width of the image
    txt              -- specifiy which dataset 1-train set 2-validation set 3-test set
    
    output:
    None, produce a series of txt files
    '''
    bbox_8 = get_absolute_gtbox(pred_box,boxs_default)
    # bbox_8 = [x1_center, y1_center, width, height, x1_min, y1_min, x1_max, y1_max]
    indices_obj = np.where(pred_confidence[:,-1]!=1)[0]
    if txt == 3:
        with open("pred/test/"+img_name+'.txt','w') as f:
            for i in indices_obj:
                class_id = np.argmax(pred_confidence[i])
                x_min = bbox_8[i,4] * width_origin
                y_min = bbox_8[i,5] * height_origin
                w = bbox_8[i,2] * width_origin
                h = bbox_8[i,3] * height_origin
                f.writelines([str(int(class_id))+' ','%.2f'%x_min+' ', '%.2f'%y_min+' ', '%.2f'%w+' ', '%.2f'%h+'\n'])
        print("pred/test/"+img_name+'.txt'+' is saved!')
    elif txt == 1 or txt == 2:
        with open("pred/train/"+img_name+'.txt','w') as f:
            for i in indices_obj:
                class_id = np.argmax(pred_confidence[i])
                x_min = bbox_8[i,4] * width_origin
                y_min = bbox_8[i,5] * height_origin
                w = bbox_8[i,2] * width_origin
                h = bbox_8[i,3] * height_origin
                f.writelines([str(int(class_id))+' ','%.2f'%x_min+' ', '%.2f'%y_min+' ', '%.2f'%w+' ', '%.2f'%h+'\n'])
        print("pred/train/"+img_name+'.txt'+' is saved!')