import torch
import torchvision
import numpy as np
import cv2
import time
import argparse

from siamese_network.models import SiameseNetwork
from vehicle_tracker.deepsort import DeepSORT
import vehicle_tracker.detect_utils as detect_utils
import config as cfg

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="2D Vehicle Tracker")
    parser.add_argument('--source', type=str, required=True, help="Source of stream. Can be a file or RTSP stream link")
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
        pretrained=True)

    model.eval().to(device)

    feature_extactor = SiameseNetwork()
    feature_extactor.load_state_dict(torch.load(cfg.feature_extractor_path))
    feature_extactor.eval().to(device)

    frame_id = 0

    cap = cv2.VideoCapture(args.source)

    deepsort = DeepSORT(feature_extactor, device)

    while True:
        start_time = time.time()
        ret, frame = cap.read()

        dets = detect_utils.predict(
            frame, model, device, 0.6, cfg.detector_input_size)

        if len(dets) != 0:
            detections, out_scores = detect_utils.get_gt(dets)

            detections = np.array(detections)
            out_scores = np.array(out_scores)
            tracker, detections_class = deepsort.run(
                frame, out_scores, detections)

            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                bbox = track.to_tlbr()  # Get the corrected/predicted bounding box
                # Get the ID for the particular track.
                id_num = str(track.track_id)

                # Draw bbox from tracker.
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(
                    bbox[2]), int(bbox[3])), (0, 0, 255), 2)
                cv2.putText(frame, str(id_num), (int(bbox[0]), int(
                    bbox[1])), 0, 5e-3 * 200, (0, 255, 0), 2)

                # Draw bbox from detector. Just to compare.
                for det in detections_class:
                    bbox = det.to_tlbr()
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(
                        bbox[2]), int(bbox[3])), (255, 255, 255), 2)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        print(f"FPS: {1/(time.time() - start_time)}")
        frame_id += 1
