import torch
import eval
import numpy as np
import os
from tqdm import tqdm

def getVideoList(datasetPath):
    """
    Args:
        datasetPath (str): The path of UCF-Crime
    Returns:
        videoList: Returns a list of video absolute paths. Length is 1900.
    """
    videoList = []
    for curDir, dirs, files in os.walk(datasetPath):
        for file in files:
            if file.endswith('.mp4'):
                # videoList.append(file)
                videoList.append(os.path.join(curDir, file))
    return videoList

def to_segments(data, num=32):
    """
    借鉴于：https://github.com/ekosman/AnomalyDetectionCVPR2018-Pytorch/blob/master/feature_extractor.py
	These code is taken from:
	https://github.com/rajanjitenpatel/C3D_feature_extraction/blob/b5894fa06d43aa62b3b64e85b07feb0853e7011a/extract_C3D_feature.py#L805
	:param data: list of features of a certain video
	:return: list of 32 segments
    """
    data = np.array(data)
    Segments_Features = []
    thirty2_shots = np.round(np.linspace(0, len(data) - 1, num=num + 1)).astype(int)
    for ss, ee in zip(thirty2_shots[:-1], thirty2_shots[1:]):
        if ss == ee:
            temp_vect = data[min(ss, data.shape[0] - 1), :]
        else:
            temp_vect = data[ss:ee, :].mean(axis=0)
	# 归一化
        temp_vect = temp_vect / np.linalg.norm(temp_vect)
        if np.linalg.norm == 0:
            logging.error("Feature norm is 0")
            exit()
        if len(temp_vect) != 0:
            Segments_Features.append(temp_vect.tolist())

    return Segments_Features


def featureWrite(saveDir, videoPath, feature):
    
    dirName = os.path.join(saveDir, videoPath.split("/")[-2])
    fileName = os.path.join(videoPath.split("/")[-1][0:-4] + ".npy")
    savePath = os.path.join(dirName, fileName)

    if not os.path.exists(dirName):
        os.mkdir(dirName)

    feature = to_segments(feature)
    np.save(savePath, feature)

    

if __name__ == "__main__":

    datasetPath = '/home/jing/project/dataset/UCF-Crime-unzip/'
    saveDir = './feature_extract/'
    videoList = getVideoList(datasetPath)

    model = eval.singleVideoProcesshhmodel()

    for videoPath in tqdm(videoList):
        pred = model.videoProcess(videoPath)
        print(pred.shape)
        featureWrite(saveDir, videoPath, pred)

    print("Feature extract done!")
   





