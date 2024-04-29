from torchvision import transforms as T
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from platform import python_version
import sys

# Function to show module versions
def show_versions():
  print("Versions...")
  print("python", python_version())
  print("torchvision", sys.modules[T.__package__].__version__)
  print("matplotlib", sys.modules[plt.__package__].__version__)
  print("cv2", cv2.__version__)
  print()
show_versions()
exit()
# Model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Classification Labels
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Function get class and box predictions
def get_prediction(img_path, threshold):
  img = Image.open(img_path)            # Load the image
  transform = T.Compose([T.ToTensor()]) # Define PyTorch Transform
  img = transform(img)                  # Apply the transform to the image
  pred = model([img])                   # Pass the image to the model
  pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]      # Prediction labels
  pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())] # Bounding boxes
  pred_score = list(pred[0]['scores'].detach().numpy())                                        # Prediction score
  pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1] # List index of scores > threshold
  pred_boxes = pred_boxes[:pred_t+1] # Top boxes
  pred_class = pred_class[:pred_t+1] # Top classes
  return pred_boxes, pred_class


# Function to show prediction result
def object_detection_api(img_path, threshold=0.5, rect_th=3, text_size=3, text_th=3):
  boxes, pred_cls = get_prediction(img_path, threshold) # Get predictions
  img = cv2.imread(img_path)                 # Read image with cv2
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB
  for i in range(len(boxes)):
    cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th) # Draw Rectangle
    cv2.putText(img,pred_cls[i], boxes[i][0],  cv2.FONT_HERSHEY_SIMPLEX, text_size, 
             (0,255,0),thickness=text_th) # Write the prediction class
  plt.figure(figsize=(20,30)) # display the output image
  plt.imshow(img)
  plt.xticks([])
  plt.yticks([])
  plt.show()

object_detection_api('./images/people.jpg', threshold=0.8)
object_detection_api('./images/car.jpg', rect_th=6, text_th=5, text_size=5)
object_detection_api('./images/traffic.jpg', rect_th=2, text_th=1, text_size=1)
object_detection_api('./images/girl_car.jpg', rect_th=15, text_th=7, text_size=5, threshold=0.8)

