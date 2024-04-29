from commons import *
from input_processing import test_image_transform
from output_processing import *
from PIL import Image, ImageDraw, ImageFont, ImageColor
import threading, subprocess, cv2, sys
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

WINDOW_NAME = "SSD DEMO"
THREAD_RUNNING = False
IMG_HANDLE = None


def draw_true_bbox(annotated_image, boxes, labels):
      
    labels = [rev_label_map[l] for l in labels[0].to('cpu').tolist()]
    
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.truetype("./calibri.ttf", 11)
    boxes = boxes.squeeze(0)
    
    for i in range(boxes.size(0)):
        text = labels[i].upper()
        box_location = boxes[i].tolist()

        draw.rectangle(xy=box_location, outline=label_color_map[labels[i]])
        draw.rectangle(xy=[l+1.0 for l in box_location], outline=label_color_map[labels[i]])

        text_size = font.getsize(text)
        text_location = [box_location[0]+2.0, box_location[1]-text_size[1]]
        textbox_location = [box_location[0], box_location[1]-text_size[1], box_location[0]+text_size[0]+4.0, box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[labels[i]])
        draw.text(xy=text_location, text=text, fill='white', font=font)
    

    del draw
    return annotated_image   


def detect_single_image(model, original_image, min_score, max_overlap, top_k, suppress=None, resize_dims=(300,300)):
    model.eval()
    image = test_image_transform(original_image, resize_dims=resize_dims).to(device)

    with torch.no_grad():
        predicted_locs, predicted_scores = model(image.unsqueeze(0))
    
    det_boxes, det_labels, det_scores = perform_nms(model.priors_cxcy, model.n_classes, predicted_locs, predicted_scores,
                                                    min_score=min_score, max_overlap=max_overlap, top_k=top_k)
    det_boxes = det_boxes[0].to('cpu')
    print("Total Objects: {}".format(det_boxes.size(0)))
    original_dims = torch.FloatTensor([original_image.width, original_image.height,
                                       original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes*original_dims
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]
    det_scores = det_scores[0].to('cpu').tolist()

    if det_labels == ['background']:
        return original_image

    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.truetype("./calibri.ttf", 11)

    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue

        print("Detected: "+str(det_labels[i].upper())+", Confidence {0:.2f}".format(100.0*det_scores[i]))
        text = det_labels[i].upper()+" {0:.2f}%".format(100.0*det_scores[i])
        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
        draw.rectangle(xy=[l+1.0 for l in box_location], outline=label_color_map[det_labels[i]])

        text_size = font.getsize(text)
        text_location = [box_location[0]+2.0, box_location[1]-text_size[1]]
        textbox_location = [box_location[0], box_location[1]-text_size[1], box_location[0]+text_size[0]+4.0, box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
        draw.text(xy=text_location, text=text, fill='white', font=font)

    del draw
    return annotated_image



def grab_image(cap):
    global THREAD_RUNNING, IMG_HANDLE
    
    while THREAD_RUNNING:
        _, IMG_HANDLE = cap.read()
        
        if IMG_HANDLE is None:
            print('grab_image(): cap.read() return None...')
            break
    THREAD_RUNNING = False

def open_window(height, width):
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, width, height)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    cv2.setWindowTitle(WINDOW_NAME, 'SSD Detection Demo')

def read_cam_and_detect(model, min_score, max_overlap, top_k, suppress=None,  resize_dims=(300,300)):
    global THREAD_RUNNING, IMG_HANDLE
    
    show_help = True
    full_screen = False
    help_text = "'Esc' to Quit, 'H' for Help, 'F' for toggling Fullscreen"
    font = cv2.FONT_HERSHEY_PLAIN
    
    while THREAD_RUNNING:
        if cv2.getWindowProperty(WINDOW_NAME, 0)<0:
            break
        
        img = IMG_HANDLE
        
        if img is not None:
            annotated_image = detect_single_image(model, Image.fromarray(img), min_score, max_overlap, top_k, suppress, resize_dims)
            cv2.imshow(WINDOW_NAME, np.array(annotated_image))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

def camera(model,  min_score, max_overlap, top_k, suppress=None, resize_dims=(300,300), height=500, width=500):
    if model is not None:
        model = model.to(device)
        model.eval()
    
    global THREAD_RUNNING
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        sys.exit("Failed to open camera!")
    
    THREAD_RUNNING = True
    th = threading.Thread(target=grab_image, args=(cap,))
    th.start()
    
    open_window(height, width)
    read_cam_and_detect(model, min_score, max_overlap, top_k, suppress, resize_dims)
    cap.release()
    cv2.destroyAllWindows()



def generate(model, dataset, n=1):
    """Generate annotated n random images from the given dataset"""
    model = model.to(device)
    model.eval()
    
    n = min(n, len(dataset))
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4*torch.cuda.device_count())
    
    for i, (image, boxes, labels, diffs) in enumerate(tqdm(loader, total=n)):
        if n == 0:
            break
        # print(boxes)
        image_name = image[0].split("/")[-1]
        image = Image.open(image[0], mode='r')
        image = image.convert('RGB')
        # print(labels)
        draw_true_bbox(image.copy(), boxes[0], labels[0]).save("./output/generated/"+image_name+"_true.jpg", "JPEG")
        detect_single_image(model, image.copy(), min_score=0.25, max_overlap=0.25, top_k=200, resize_dims=(500,500)).save("./output/generated/"+image_name+"_predicted.jpg", "JPEG")
        n-=1