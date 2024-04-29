from model import YOLO
from utils import *


def main():

    # Read image
    image = cv2.imread("image.jpg")
    # Preprocess image
    processed_image = preprocess_image(image)
    # Load classes
    file = open("data/coco.names", "r")
    classes = [line.strip() for line in file.readlines()]
    # Load colors
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    # Load Model
    model = YOLO("config/yolo.cfg")
    model.load_weights("config/yolov3.weights")
    model.eval()
    # Prediciton
    predictions = model(processed_image)
    # Non-maximum suppression
    prediction = NMS(predictions)[0]
    if prediction is not None:
        # Rescale bouding box
        prediction[:, :4] = rescale(prediction[:, :4], image.shape[:2])
        for x1, y1, x2, y2, obj, confidence, label in prediction:
            # Get color of label
            color = colors[int(label)]
            # Draw bouding Box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            # Draw Text
            cv2.putText(image, f"{classes[int(label)]}: {confidence:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("image", image)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
