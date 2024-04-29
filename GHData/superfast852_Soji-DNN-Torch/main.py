import torch  # For Model
import cv2  # For Cam
import time  # Inference measuring
import local_utils.Label_xtr as Label_xtr # Custom Label Extractor TODO: Develop Extractor functions

# Parameter Wall TODO: Add argparse
debug = False  # Enable Function Outputs
one = False
model_type = "v7_custom"  # Select from: v5s, v5m, v5n, v7, v7t, v5_custom, v7_custom
v5_custom_path = "C:/Users/GG/Desktop/Code/ML/Models/v5s_70/weights/best.pt"  # Path to a YOLOv5 model trained on a custom dataset
v7_custom_path = "prebuilts/custom/v7t_p5hyps_aqua/V1-416-86%/yolov7.pt"  # Path to a YOLOv7 model trained on a custom dataset
custom_yaml = "prebuilts/custom/v7t_p5hyps_aqua/data.yaml"  # Path to the custom dataset's data.yaml file
src = "Testing/testvid.mp4"
min_conf = 0.5
avg_fps = []  # List for Calculating Average FPS at ending


def infer(frame, model, debug=False):

    results = model(frame)  # infer
    # extract and convert results to list
    dets = results.xyxy[0].tolist()
    if debug: print(dets)
    return dets


def draw(dets, frame, classes, debug=True):
    xy1 = (int(dets[i][0]), int(dets[i][1]))  # top left point
    xy2 = (int(dets[i][2]), int(dets[i][3]))  # bottom right point
    conf, lbl = dets[i][4:]  # confidence and output label

    cv2.rectangle(frame, xy1, xy2, (0, 255, 0), 3, cv2.LINE_AA)  # draw rectangle
    cv2.putText(frame, f"{classes[int(lbl)]}: {conf:.2f}", xy1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # write label and conf atop rectangle
    if debug:
        print(f"Detection {i + 1}: \n   BBox: {xy1, xy2} \n   Label: {classes[int(lbl)]} \n   Confidence: {conf:.2f}")  # output info
        cx = (dets[i][2] + dets[i][0]) / 2  # Center of X
        cy = (dets[i][3] + dets[i][1]) / 2  # Center of Y
        cv2.circle(frame, (int(cx), int(cy)), 5, color=(0, 0, 255), thickness=-1)


def draw_1(dets, frame, classes, debug=True):
    xy1 = (int(dets[0]), int(dets[1]))  # top left point
    xy2 = (int(dets[2]), int(dets[3]))  # bottom right point
    conf, lbl = dets[4:]  # confidence and output label

    cv2.rectangle(frame, xy1, xy2, (0, 255, 0), 3, cv2.LINE_AA)  # draw rectangle
    cv2.putText(frame, f"{classes[int(lbl)]}: {conf:.2f}", xy1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # write label and conf atop rectangle
    if debug:
        print(f"Detection: \n   BBox: {xy1, xy2} \n   Label: {classes[int(lbl)]} \n   Confidence: {conf:.2f}")  # output info
        cx = (dets[2] + dets[0]) / 2  # Center of X
        cy = (dets[3] + dets[1]) / 2  # Center of Y
        cv2.circle(frame, (int(cx), int(cy)), 5, color=(0, 0, 255), thickness=-1)


def pos_servo(dets, frame):
    #from adafruit_servokit import ServoKit
    #pca = ServoKit(channels=8)
    detX = (dets[2] + dets[0]) / 2  # Center of X
    detY = (dets[3] + dets[1]) / 2  # Center of Y
    ccy, ccx = [x/2 for x in frame.shape[:2]]

    if ccx<detX-50: xServ = '-'
    elif ccx>detX+50: xServ = '+'
    else: xServ = "X axis is Centered"

    if ccy<detY-50: yServ = '-'
    elif ccy>detY+50: yServ = '+'
    else: yServ = "Y axis is Centered"

    return [xServ, yServ]


def frame_debug(frame):
    cy, cx = frame.shape[:2]
    ccx = int(cx/2)
    ccy = int(cy/2)
    cv2.rectangle(frame, (ccx-50, ccy-50), (ccx+50, ccy+50), (255, 0, 0), 5, cv2.LINE_AA)

    cv2.line(frame, (0, ccy), (ccx-50, ccy), (0,0,255), 2, cv2.LINE_AA) #left line
    cv2.line(frame, (ccx, 0), (ccx, ccy-50), (0, 0, 255), 2, cv2.LINE_AA) #upper line
    cv2.line(frame, (ccx+50, ccy), (cx, ccy), (0, 0, 255), 2, cv2.LINE_AA) #right line
    cv2.line(frame, (ccx, ccy+50), (ccx, cy), (0, 0, 255), 2, cv2.LINE_AA) #lower line


if __name__ == "__main__":
    try:
        if model_type == 'v5m':
            model = torch.hub.load("ultralytics/yolov5", 'custom', "prebuilts/yolov5m.pt", verbose=debug)

        elif model_type == 'v5s':
            model = torch.hub.load("ultralytics/yolov5", 'custom', "prebuilts/yolov5s.pt", verbose=debug)

        elif model_type == 'v5n':
            model = torch.hub.load("ultralytics/yolov5", 'custom', "prebuilts/yolov5n.pt", verbose=debug)

        elif model_type == "v5_custom":
            model = torch.hub.load('ultralytics/yolov5', 'custom', v5_custom_path, verbose=debug)  # YoloV7 Tiny Model

        elif model_type == 'v7':
            model = torch.hub.load('C:/Users/GG/Desktop/Code/ML/yolov7', 'custom', "prebuilts/yolov7.pt", source='local', verbose=debug)  # YoloV7 Tiny Model

        elif model_type == 'v7t':
            model = torch.hub.load('C:/Users/GG/Desktop/Code/ML/yolov7', 'custom', "prebuilts/yolov7t.pt", source='local', verbose=debug)  # YoloV7 Tiny Model

        elif model_type == "v7_custom":
            model = torch.hub.load('C:/Users/GG/Desktop/Code/ML/yolov7', 'custom', v7_custom_path, source='local', verbose=debug)  # YoloV7 Tiny Model

        # Extract classes from file
        if not model_type.endswith("custom"):
            classes = Label_xtr.getLabelsFromTxt(path="coco-lbl.txt", verbose=debug)  # Coco Labels Extract
        else:
            classes = Label_xtr.getLabelsFromYaml(custom_yaml, verbose=debug)  # AquaTrash Labels
            #classes = Label_xtr.getLabelsFromYaml(custom_yaml, verbose=debug)  # Trash-Filter Labels
            #classes = ['-', 'cardboard', 'glass', 'metal', 'plastic'] #Manual Labels

        # Open Webcam
        vid = cv2.VideoCapture(src)
        while vid.isOpened():
            start_time = time.time()
            ret, frame = vid.read()  # read webcam image info
            detections = infer(frame, model, debug=debug)  # goto infer function atop

            confs = []
            for i in range(len(detections)):
                # Show Webcam Image with Detection Info
                if not one:
                    if detections[i][4] >= min_conf:
                        draw(detections, frame, classes, debug=debug)
                confs.append(detections[i][4])

            if confs:
                most_conf_det = confs.index(max(confs))
                if one: draw_1(detections[most_conf_det], frame, classes, debug=debug)

                if debug:
                    mcd_frame_adj = pos_servo(detections[most_conf_det], frame)
                    if mcd_frame_adj == ['+', '+']:
                        print("Upper Left")
                    elif mcd_frame_adj == ['-', '-']:
                        print("Lower Right")
                    elif mcd_frame_adj == ['+', '-']:
                        print("Lower Left")
                    elif mcd_frame_adj == ['-', '+']:
                        print("Upper Right")
                    else:
                        print(mcd_frame_adj)
                    print(f"{detections[most_conf_det][4]:.3f}: {classes[int(detections[most_conf_det][5])]}")
                    frame_debug(frame)

            cv2.imshow("Model Modified Output", frame)
            # FrameRate calculations
            if debug:
                fps = 1/(time.time()-start_time)
                avg_fps.append(fps)
                print(f"FPS: {fps:.3f}")

            if cv2.waitKey(1) & 0xFF == ord('q'): break  # stop process if q is pressed

    except:
        if debug: print(f"Average FrameRate: {sum(avg_fps)/len(avg_fps):.3f}")
        vid.release()
        cv2.destroyAllWindows()
