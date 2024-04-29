from torchvision import transforms
from utils import *
from PIL import Image, ImageDraw, ImageFont
import glob
import os
import numpy as np
import cv2
from os.path import isfile, join
import glob
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from utils_package.online_tubes import VideoPostProcessor
#device= torch.device("cpu")

# Load model checkpoint
checkpoint = 'BEST_checkpoint_ssd300.pth.tar'
checkpoint = torch.load(checkpoint)
start_epoch = checkpoint['epoch'] + 1
best_loss = checkpoint['best_loss']
print('\nLoaded checkpoint from epoch %d. Best loss so far is %.3f.\n' % (start_epoch, best_loss))
model = checkpoint['model']
model = model.to(device)
model.eval()

# Transforms
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


COLOR_WHEEL = ('red', 'blue', 'brown', 'darkblue', 'green',
               'darkgreen', 'brown', 'coral', 'crimson', 'cyan',
               'fuchsia', 'gold', 'indigo', 'red', 'lightblue',
               'lightgreen', 'lime', 'magenta', 'maroon', 'navy',
               'olive', 'orange', 'orangered', 'orchid', 'plum',
               'purple', 'tan', 'teal', 'tomato', 'violet')


def detect(original_image, min_score, max_overlap, top_k, suppress=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """

    # Transform
    image = normalize(to_tensor(resize(original_image)))

    # Move to default device
    image = image.to(device)

    # Forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)

    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    if det_labels == ['background']:
        # Just return original image
        return original_image

    # Annotate
    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.load_default()

    # Suppress specific classes, if needed
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue

        # Boxes
        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
            det_labels[i]])  # a second rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 2. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a third rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 3. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a fourth rectangle at an offset of 1 pixel to increase line thickness

        # Text
        text_size = font.getsize(det_labels[i].upper())
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                            box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
        draw.text(xy=text_location, text=det_labels[i].upper(), fill='white',
                  font=font)
    del draw

    return annotated_image


def detect1(original_image,idx,target_dets,target_scores):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """
    
    model.eval()
    # Transform
    image = normalize(to_tensor(resize(original_image)))

    # Move to default device
    image = image.to(device)

    # Forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    with torch.no_grad():
        # Detect objects in SSD output
        det_boxes, det_scores = model.detect_objects1(predicted_locs, predicted_scores)

    # Move detections to the CPU
    det_boxes = det_boxes.to('cpu')
    det_scores=det_scores.to('cpu')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims
    
    num_box=det_boxes.size(0)


    target_dets[idx,1,:num_box,:]=det_boxes.numpy()
    target_scores[idx,1,:num_box,:]=det_scores.numpy()
        
    if idx!=0:
        target_dets[idx,0,:,:]=target_dets[idx-1,1,:,:]
        target_scores[idx,0,:,:]=target_scores[idx-1,1,:,:]
    else:
        target_dets[idx,0,:,:]=target_dets[idx,1,:,:]
        target_scores[idx,0,:,:]=target_scores[idx,1,:,:]


    return target_dets,target_scores

def convert_frames_to_video(video_dataset, fps):
    print("Converting...")

    # define save dir
    output_dir = "./ouput_videos/outdoor1"
    pathIn=output_dir+'/'
    pathOut=os.path.join(output_dir, 'test.avi')
    

    
    frame_array = []
    files = sorted(glob.glob(os.path.join(pathIn, "*.png")))
    size = (0,0)
    
    for i in range(len(files)):
        filename=files[i]
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        print(filename)
        #inserting the frames into an image array
        frame_array.append(img)
 
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
 
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()
    
def visualize_with_paths(video_dataset, video_post_proc,imagenet_vid_classes):

    print("Visualizing...")

    # define save dir
    
    output_dir ="./ouput_videos/outdoor1"
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir)

    det_classes = imagenet_vid_classes

    num_frames = len(video_dataset)

    for i_frame in range(num_frames):
        print('frame: {}/{}'.format(i_frame, num_frames))
        fig, ax = plt.subplots(figsize=(12, 12))
        disp_image = video_dataset[i_frame]
        for i_pth, cls_ind in enumerate(video_post_proc.path_labels): # iterate over path labels
            cls_ind = int(cls_ind)
            ax.imshow(disp_image, aspect='equal')
            class_name = det_classes[cls_ind]
            path_starts =  video_post_proc.path_starts[i_pth]
            path_ends = video_post_proc.path_ends[i_pth]
            if i_frame >= path_starts and i_frame <= path_ends: # is this frame in the current path
                # bboxes for this class path
                bbox = video_post_proc.path_boxes[i_pth][i_frame-path_starts].cpu().numpy() 
                # scores for this class path
                score = video_post_proc.path_scores[i_pth][i_frame-path_starts].cpu().numpy() 
                
                ax.add_patch(
                        plt.Rectangle((bbox[0], bbox[1]),
                            bbox[2] - bbox[0],
                            bbox[3] - bbox[1], fill=False,
                            edgecolor=COLOR_WHEEL[cls_ind], linewidth=3.5)
                        )
                ax.text(bbox[0], bbox[1] - 2,
                        '{:s} {:.3f}'.format(class_name, score[0]),
                        bbox=dict(facecolor=COLOR_WHEEL[cls_ind], alpha=0.5),
                        fontsize=14, color='white')

        plt.axis('off')
        plt.tight_layout()
        #plt.show()
        im_save_name = os.path.join(output_dir,"%#09d.png" % (i_frame))
        print('Image with bboxes saved to {}'.format(im_save_name))
        plt.savefig(im_save_name)
        plt.clf()
        plt.close('all')





if __name__ == '__main__':
#    img_path = './data/VOC2007/JPEGImages/000001.jpg'
#    img_path="/home/fengy/Documents/tiny-faces-pytorch_changed/data/WIDER/WIDER_test/images/0--Parade/0_Parade_marchingband_1_9.jpg"
    input_path='/home/fengy/Documents/pytorch-detect-to-track_1/output_videos/outdoor_video_processed_imagevid+det_res50_epoch2_thre=0.2_fps=25'
#    input_path="/home/fengy/Downloads/ILSVRC2015_VID/ILSVRC2015/Data/VID/train/VIRAT-V1/VIRAT_S_000001"
    videos = sorted(glob.glob(os.path.join(input_path, "*.png")))

    images=[]
    imagenet_vid_classes = ['background','aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    num_images=len(videos)
    target_dets=np.zeros(shape=(num_images,2,10000,4))
    target_scores=np.zeros(shape=(num_images,2,10000,len(imagenet_vid_classes)))
    
    for index, frame in enumerate(videos):
        original_image = Image.open(frame, mode='r')
        original_image = original_image.convert('RGB')
        images.append(original_image)
        target_dets,target_scores=detect1(original_image,index,target_dets,target_scores)
    
    vid_pred_boxes=torch.FloatTensor(target_dets)
    vid_scores=torch.FloatTensor(target_scores)
    vid_post_proc = VideoPostProcessor(vid_pred_boxes, vid_scores, imagenet_vid_classes)
    paths = vid_post_proc.class_paths(path_score_thresh=0.05)

    if vid_post_proc.path_total_score.numel() > 0:
           	  visualize_with_paths(images, vid_post_proc,imagenet_vid_classes)
           	  fps = 10
           	  convert_frames_to_video(images,fps)
    else:
           print("No object had been deteced!")
