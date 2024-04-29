import os
import shutil
def rename_segmentation_annotations():
    annotations = 'data_segmentation/annotations/'
    ans = os.listdir(annotations)
    for i in ans:
        c = i.replace('_refined_mask', '')
        os.rename(annotations + '/' + i, annotations+'/'+c)

def rearange_detection_data():
    detection_dataset_path = '/data_detection/'
    test_path = detection_dataset_path + 'test_zip/test/'
    train_path = detection_dataset_path + 'train_zip/train/'

    test_files = os.listdir(test_path)
    train_files = os.listdir(train_path)

    all_files = []

    for  train in  train_files:

        if train[-3:] == 'jpg':
            shutil.move(train_path + train, detection_dataset_path + 'images')
        else :
            shutil.move(train_path + train, detection_dataset_path + 'annotations')


