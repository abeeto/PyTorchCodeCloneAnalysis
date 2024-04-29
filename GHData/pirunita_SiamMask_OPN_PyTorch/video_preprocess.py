
import argparse
import logging
import os

import cv2

parser = argparse.ArgumentParser(description='VideoPreprocessing')

parser.add_argument('--name', type=str, default='VideoPreprocessing')
parser.add_argument('--base_path', default='/data/tennis', help='datasets')
parser.add_argument('--f', type=float, default=0.25)
args = parser.parse_args()

# Set logger
logger = logging.getLogger('MLP')
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)

extension_list = ['.mov', '.MOV', '.MP4']

if __name__ == '__main__':
    file_list = sorted(os.listdir(os.path.join(args.base_path)))
    file_name = file_list[0]
    file_extension = os.path.splitext(file_name)[1]
    
    count_num = 0
    if file_extension in extension_list:
        logger.info('We found video file, so we will encode video to image files per frame!!') 
        video_capture = cv2.VideoCapture(os.path.join(args.base_path, file_name))
        
        while(video_capture.isOpened()):
            ret, frame = video_capture.read()
            resized_frame = cv2.resize(frame, dsize=(0, 0), fx=args.f, fy=args.f, interpolation=cv2.INTER_AREA)
            converted_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(args.base_path, '{:05d}.jpg'.format(count_num)), resized_frame)
            count_num += 1
            cv2.imshow('Encoding... if you want to stop, press q', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        video_capture.release()
        cv2.destroyAllWindows()

        logger.info("Video preprocessing finished ! in %s" %(os.path.join(args.base_path)))
    else:
        logger.info('There is no video file !!!')
    
        
    