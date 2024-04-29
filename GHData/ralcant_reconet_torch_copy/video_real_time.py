########### Imports from avatarify, not needed for now... #################
# import os, sys
""" from sys import platform as _platform
import glob
import yaml
import time
import requests

# import numpy as np
# import cv2

from avatarify.afy.videocaptureasync import VideoCaptureAsync
from avatarify.afy.arguments import opt
from avatarify.afy.utils import info, Once, Tee, crop, pad_img, resize, TicToc
import avatarify.afy.camera_selector as cam_selector

log = Tee('./avatarify/var/log/cam_fomm.log')
print = log
def select_camera(config):
    cam_config = config['cam_config']
    cam_id = None

    if os.path.isfile(cam_config):
        with open(cam_config, 'r') as f:
            cam_config = yaml.load(f, Loader=yaml.FullLoader)
            cam_id = cam_config['cam_id']
    else:
        cam_frames = cam_selector.query_cameras(config['query_n_cams'])

        if cam_frames:
            cam_id = cam_selector.select_camera(cam_frames, window="CLICK ON YOUR CAMERA")
            log(f"Selected camera {cam_id}")

            with open(cam_config, 'w') as f:
                yaml.dump({'cam_id': cam_id}, f)
        else:
            log("No cameras are available")

    return cam_id
 """
####################################################
import datetime
import cv2
import os
import argparse
import torch
# import torchvision
import numpy as np
from network import ReCoNet
import sys
import pyaudio 
import wave
from audio_handler import VideoPlay
from model_creator import get_all_models
from arguments import opt

def video_real_time( video_source_path, using_local_video, device, use_audio, audio_path=None):#, output_dir='output.avi', fps=30, concat=False):
    #assert os.path.exists(video_source_path), f"Video path {video_source_path}, doesnt exist!"
    vid_obj = cv2.VideoCapture(video_source_path)
    #### Audio Processing ######
    if use_audio:
       audio_controler = VideoPlay(
           video_path=video_source_path,
           audio_path= audio_path,
           nframes = vid_obj.get(cv2.CAP_PROP_FRAME_COUNT),
           spf = 1/(vid_obj.get(cv2.CAP_PROP_FPS))
       )
    ##########
    # If w and h are small enough, then the process will be faster
    w  = 240#480 #int(vid_obj.get(cv2.CAP_PROP_FRAME_WIDTH) /3 ) # dividing because we will be resizing!
    h = 180#360 #int(vid_obj.get(cv2.CAP_PROP_FRAME_HEIGHT) /3 )# float
    img_size = (w, h)
    total_frame_count = 0
    frame_count_fps = 0
    process_time = 0
    print(f"img_shape = {img_size}")


    # fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    # video_writer = cv2.VideoWriter(video_dir, fourcc, fps, img_size)

    start = True
    curr_model = None
    models, originals = get_all_models(device)
    curr_model = models[0]
    print("models trained!")
    start_time = datetime.datetime.now()
    try:
        while True:
            key = cv2.waitKey(1)
            #if key != -1:
            #    print(f"key = {key}")
            if key == ord('s'): #Press s = start!
                start = True
                start_time=datetime.datetime.now()
                frame_count_fps = total_frame_count
                print("starting...")
            if key == ord('q'): #Press q = quit!
                print("Quitting!")
                break
            if 48 < key < 58: #Press 1-9: Change between models!
                curr_model = (key-49) % len(models) #min(key - 49, len(models) - 1)
                print(f"Pressed key {key-48}, loading model #{key-48}")
                curr_model = models[curr_model]
            if key == 48:     #Press 0 = original frame
                print("Showing original")
                curr_model = None
            if key == ord('p'):
                print("Pausing... press S to start!")
                start = False
                process_time += (datetime.datetime.now()-start_time).total_seconds()
            if start:
                success, image = vid_obj.read() 
                if not success: #Finished video!
                    break
                frame = cv2.resize(image, dsize=(w, h))#,fx=1/3, fy=1/3)
                if curr_model is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    img = torch.from_numpy(frame.astype('float32') / 255.0).permute(2, 0, 1)
                    img = img.to(device)
                    _, output = curr_model(img.unsqueeze(0))
                    concat_img = output.squeeze(0).permute(1, 2, 0)
                    # if concat:
                    #     concat_img = torch.cat([img, output.squeeze(0)], dim=1).permute(1, 2, 0)
                    concat_cv2 = concat_img.detach().cpu().numpy()

                    frame = concat_cv2 * 255
                    frame = frame.astype('uint8')
                    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, dsize=(640, 360))
                cv2.imshow("video!", frame)
                if use_audio:
                   audio_controler.play_audio_for_frame(total_frame_count)
                total_frame_count+=1

                log_fps = False # This might affect the speed of the process..?
                if log_fps:
                   fps = (total_frame_count-frame_count_fps)/(datetime.datetime.now()-start_time).total_seconds()
                   print(f"FPS = {fps}")
        # video_writer.release()
    except KeyboardInterrupt:
        print("Keyboard Interrupt!")
    process_time += (datetime.datetime.now()-start_time).total_seconds() #To take into account frames before pause
    if use_audio:
       audio_controler.close()
    #print(f"Video duration: {5}s")
    #print(f"Process duration (not counting pauses): {process_time}")
    cv2.destroyAllWindows()
    # end_time=datetime.datetime.now()
    # print('end time:',end_time.strftime('%Y/%m/%d %H:%M:%S'))
    # # print(f'FPS = {frame_count/(end_time-start_time).total_seconds()}')
    # print('cost time:',end_time-start_time)
    # # print('the video location is:', video_dir)
    # print('finish')

if __name__ == "__main__":
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # with open('avatarify/config.yaml', 'r') as f:
    #     config = yaml.load(f, Loader=yaml.FullLoader)
    # global display_string
    # display_string = ""
    # log('Starting process..')
    # predictor_args = {
    #     'config_path': opt.config,
    #     'checkpoint_path': opt.checkpoint,
    #     'relative': opt.relative,
    #     'adapt_movement_scale': opt.adapt_scale,
    #     'enc_downscale': opt.enc_downscale
    # }
    # if opt.is_worker:
    #     log("opt.is_worker")
    #     sys.exit(0)
    #     # from avatarify.afy import predictor_worker
    #     # predictor_worker.run_worker(opt.in_port, opt.out_port)
    #     # sys.exit(0)
    # elif opt.is_client:
    #     log("opt.is_client")
    #     from avatarify.afy import predictor_remote
    #     try:
    #         predictor = predictor_remote.PredictorRemote(
    #             in_addr=opt.in_addr, out_addr=opt.out_addr,
    #             **predictor_args
    #         )
    #     except ConnectionError as err:
    #         log(err)
    #         sys.exit(1)
    #     log("Predictor initialized!")
    #     predictor.start()
    #     log("Predictor started...")
    # else:
    #     log("not opt.is_client and not opt.is_worker")
    #     sys.exit(0)
    #     # from avatarify.afy import predictor_local
    #     # predictor = predictor_local.PredictorLocal(
    #     #     **predictor_args
    #     # )
    # log("Trying to select camera")
    # cam_id = select_camera(config)

    # if cam_id is None:
    #     log("exiting...")
    #     exit(1)
    # log("continuing...")

    print(opt.use_audio)
    if opt.mode == "existing_video":
        video_name = opt.video_source
        video_path = f"videos/{video_name}.mp4"
        audio_path = f"audios/{video_name}.wav"
        video_real_time(video_source_path= video_path, using_local_video=True, device=device, use_audio= opt.use_audio, audio_path=audio_path)
    elif opt.mode == "video_stream":
        video_name = opt.video_source
        #video_path = f"videos/{video_name}.mp4"
        camera = 0
        audio_path = f"audios/{video_name}.wav"
        video_real_time(video_source_path = camera, using_local_video=False, device=device, use_audio=opt.use_audio)
    else:
        print(f"mode: {opt.mode} is not supported here. See video_cv2.py for other modes. Use mode existing_video or video_stream")
