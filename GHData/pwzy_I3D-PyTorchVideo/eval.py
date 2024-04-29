from os import pread
import torch
import json
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
)
from typing import Dict
import cv2
import numpy as np

class VideoIter(torch.utils.data.Dataset):
    def __init__(self, videoPath, videoClipSec, transform) -> None:
        super().__init__()
        
        self.videoClipSec = videoClipSec
        self.video = EncodedVideo.from_path(videoPath)
        self.transform = transform

    def __len__(self):
        return len(self.videoClipSec)

    def __getitem__(self, index: int):
        video_data = self.video.get_clip(start_sec=self.videoClipSec[index][0], end_sec=self.videoClipSec[index][1]) # [3, 16, 240, 320]
        video_data = self.transform(video_data)
        # print(video_data['video'].shape)
        return video_data['video']

class singleVideoProcesshhmodel():
    def __init__(self) -> None:
        self.device = self.deviceSelect()
        self.model = self.loadModel(self.device)

        # 视频处理参数设置
        self.side_size         = 224 # 设置要缩放的短边大小
        self.mean              = [0.45, 0.45, 0.45]
        self.std               = [0.225, 0.225, 0.225]
        self.crop_size         = 224 # 要裁剪的大小
        self.clip_frames        = 16 # 采样多少帧
        self.sampling_rate     = 1   # 每几帧采一次样（采样频率）
        self.frames_per_second = 30  # 视频帧率

        self.transform = self.videoTransform()
        

    def videoProcess(self, videoPath):
        videoCapture=cv2.VideoCapture(videoPath)                       # 获取视频对象
        framesNum = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)              # 视频帧总数
        clipNum = int(framesNum // (self.clip_frames * self.sampling_rate)) # 总的clip数

        step = np.linspace(0, clipNum*(self.clip_frames * self.sampling_rate), clipNum + 1) # 产生clip总数+1个数作为step
        step_sec = step / self.frames_per_second  # 转化为秒数

        videoClipSec = []
        for k in range(clipNum):
            startSec = step_sec[k]
            endSec = step_sec[k+1]
            videoClipSec.append([startSec, endSec])

        dataset = VideoIter(videoPath, videoClipSec, self.transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)
        
        videoFeature = []
        for clip in dataloader:
            # print(clip.shape)
            clip = clip.to(self.device)
            pred = self.model(clip)
            videoFeature.append(pred.detach().cpu())
        return torch.cat(videoFeature, dim=0)

    ################################################################################
    # Define device
    ################################################################################
    def deviceSelect(self):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ################################################################################
    # Load Model
    # Reference: https://pytorchvideo.org/docs/tutorial_torchhub_inference
    ################################################################################
    def loadModel(self, device):
        model_name = "i3d_r50"
        model = torch.hub.load("facebookresearch/pytorchvideo", model=model_name, pretrained=True)
        # 去掉类别的转换
        model.blocks[6].proj = torch.nn.Sequential()
        model = model.to(device)
        model = model.eval()

        return model

    ################################################################################
    # Video transform
    ################################################################################
    def videoTransform(self):
        ########################################
        # Define transform
        ########################################
        transform =  ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(self.clip_frames),
                    Lambda(lambda x: x/255.0),
                    NormalizeVideo(self.mean, self.std),
                    ShortSideScale(
                        size=self.side_size
                    ),
                    CenterCropVideo(self.crop_size),
                ]
            ),
        )

        return transform 

if __name__ == "__main__":
    model = singleVideoProcesshhmodel()
    pred = model.videoProcess("/home/jing/project/dataset/UCF-Crime-unzip/Training-Normal-Videos-Part-1/Normal_Videos480_x264.mp4")
    print(pred.shape)


