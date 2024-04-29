import cv2
import glob
import os
def video_preprocessing_trim(src_folder_dir,dst_folder_dir,roi,video_suffix = "avi"):
    if not os.path.exists(dst_folder_dir):
        os.makedirs(dst_folder_dir)

    video_file_dirs = glob.glob(os.path.join(src_folder_dir,"*"+video_suffix))
    
    for video_file_dir in video_file_dirs:
        video_file_name = os.path.basename(video_file_dir)

        src_video = cv2.VideoCapture(video_file_dir)

        fps = src_video.get(cv2.CAP_PROP_FPS)
        frame_size = (int(roi[2]-roi[0]), int(roi[3]-roi[1]))
        show_result_video_dir = os.path.join(dst_folder_dir,video_file_name)
        dst_video = cv2.VideoWriter(show_result_video_dir,cv2.VideoWriter_fourcc(*'DIVX'),fps,frame_size)
        
        success,src_frame = src_video.read()

        while success:
            dst_frame = src_frame[roi[1]:roi[3],roi[0]:roi[2]]
            dst_video.write(dst_frame)
            success,src_frame = src_video.read()
'''
src_folder_dir="/data2/qilei_chen/jianjiwanzhengshipin2/xiangyachangde/"
dst_folder_dir="/data2/qilei_chen/jianjiwanzhengshipin2/preprocessed/"
roi = [441, 1, 1278, 720]
video_preprocessing_trim(src_folder_dir,dst_folder_dir,roi)
'''
src_folder_dir="/home/cql/Downloads/issue_videos/original_videos/"
dst_folder_dir="/home/cql/Downloads/issue_videos/processed_videos/"
roi = [156, 40, 698, 527]
video_preprocessing_trim(src_folder_dir,dst_folder_dir,roi,'avi')