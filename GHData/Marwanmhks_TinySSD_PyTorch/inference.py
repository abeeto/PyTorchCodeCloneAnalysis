from utils.tools import *
from model.tiny_ssd import Tiny_SSD

# ---------------------------------------------------------
#                   Drawing function
# ---------------------------------------------------------
def showres(_image, _output, _threshold, _name_classes):
    """ Display image and bounding box
    :param _image: PIL Image
    :param _output: (chose, 10) actual image size
    :param _threshold: confidence threshold
    :param _name_classes: list of classes
    :return:
    """
    plt.cla() # Clear the previous moment
    plt.draw() # will refresh the drawing
    fig = plt.imshow(_image)
    for row in _output:
        score = float(row[1])
        _class = _name_classes[row[0].long()]
        if score < _threshold: continue
        bbox = [row[2:6]]
        abox = [row[6:10]]
        # show_bboxes(fig.axes, abox, f'{_class}[{score:.2f}]', 'w') # show the corresponding anchors
        show_bboxes(fig.axes, bbox, f'{_class}[{score:.2f}]', 'b') # '%.2f'%value formatted output


"""
# ---------------------------------------------------------
#                     Camera
# ---------------------------------------------------------
import cv2
cap = cv2.VideoCapture(0)
ssd = Tiny_SSD()
name_classes, num_classes = ssd.name_classes, ssd.num_classes
device = ssd.device

try:
    while True:
        _, frame = cap.read() # h w c
        frame[:] = frame[..., [2, 1, 0]] # cv2 images are all BGR and need to be converted to RGB
        frame = Image.fromarray(frame, mode='RGB')
        output = ssd.inference(frame)
        showres(frame, output, 0.3, ssd.name_classes)
        plt.pause(0.001)
except KeyboardInterrupt:
    cap.release()
"""

# ---------------------------------------------------------
#                     Picture
# ---------------------------------------------------------

image = Image.open('C:\\Users\Marwan\PycharmProjects\TinySSD_Banana\TinySSD_Banana\VOCdevkit\\test.jpg').convert('RGB')
ssd = Tiny_SSD()
output = ssd.inference(image).to('cpu')
showres(image, output, 0.4, ssd.name_classes)
plt.pause(30)
