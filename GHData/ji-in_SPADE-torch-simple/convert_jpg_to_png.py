import os
import glob
from PIL import Image

path = './dataset/spade_celebA/image'

# jpg파일을 저장하기 위한 디렉토리의 생성
if not os.path.exists(path+'_png'):
    os.mkdir(path+'_png') 

# 모든 png 파일의 절대경로를 저장
all_image_files=glob.glob(path+'/*.jpg') 

for i, file_path in enumerate(all_image_files):                   # 모든 png파일 경로에 대하여
    img = Image.open(file_path).convert('RGB')  # 이미지를 불러온다.

    directories=file_path.split('/')                # 절대경로상의 모든 디렉토리를 얻어낸다.
    directories[-2]+='_png'                     # 저장될 디렉토리의 이름 지정
    directories[-1]=directories[-1][:-4]+'.png'  # 저장될 파일의 이름 지정
    save_filepath='/'.join(directories)          # 절대경로명으로 바꾸기
    img.save(save_filepath, quality=100)       # jpg파일로 저장한다.
    print(f'{i}-th iteration save!')