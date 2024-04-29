import cv2
import os 
import argparse


def video_to_frame(filename):
     
    # vid_obj = cv2.VideoCapture(f"{general_videos_folder}{inputs_subfolder}{filename}.mp4")
    vid_obj = cv2.VideoCapture(f"videos/{filename}.mp4")
    os.makedirs(f"frames/{filename}/", exist_ok=True)
    get_frame_path = lambda part: f"frames/{filename}/{part}.jpg"
    count = 0
    print(f"Total # of frames according to code {int(vid_obj.get(cv2.CAP_PROP_FRAME_COUNT))}")
    fps = vid_obj.get(cv2.CAP_PROP_FPS)
    print(f"Frames per second = {fps}")
    print("Now reading the frames... [This might take a while]")
    while True: 
        success, image = vid_obj.read() 
        if not success:
            break
         
        image = cv2.resize(image, dsize=(640, 360))#fx=1/3, fy=1/3)
        # Saves the frames with frame-count 
        cv2.imwrite(get_frame_path(count), image) 
        count += 1
    print(f"Total # frames accroding to loop = {count}")
    # print(f"Succesfully wrote all the frames in the folder {general_videos_folder}{frames_subfolder}{filename}/")
    return fps
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert video to frames')
    parser.add_argument('--filename', type=str, help="Name of the video in the videos/ folder (without the extension)")
    args = parser.parse_args()
    video_to_frame(args.filename)
