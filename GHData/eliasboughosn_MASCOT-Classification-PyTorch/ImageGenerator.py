#Library imports
import pandas as pd
import cv2

#dataframe
df = pd.read_csv("/home/eboughos/sex_images/annotations/vishal13_df.txt",
                 sep=" ", names=['FRAME','ID','SEX','X1','Y1','X2','Y2'])

df=df[df.ID.isin([0,1,2,3,4,5,6,7,8,9,10,11,12])]
df = df[(df.SEX != "l")]

# Loading video
path_vid="./../../../data/stars/user/vpani/MASCOT/All_Data/Density/PR002/2104_PR002_Low_13.MOV"
video = cv2.VideoCapture(path_vid)
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
success, image = video.read()
counter = 0
step = 0
print(total_frames)
print(success)

while success:
    df_temp = df[df.FRAME == counter]
    print("df_temp is: ",len(df_temp))
    for insect_num in range(len(df_temp)):
        X1 = df_temp.iloc[insect_num].X1
        X2 = df_temp.iloc[insect_num].X2
        Y1 = df_temp.iloc[insect_num].Y1
        Y2 = df_temp.iloc[insect_num].Y2
        SEX = df_temp.iloc[insect_num].SEX
        ROI = image[Y1:Y2,X1:X2]
        #print('write  ----IMAGE---- frame{}--- insectNum{}: {}_{}_{}_{} '.format(counter,insect_num,X1,Y1,X2,Y2), success)
        #cv2.imwrite(r"/data/stars/user/eboughos/sex/all/PR/13frame{}_insectNum{}_sex{}_{}_{}_{}_{}.jpg".format(
        counter,insect_num,SEX, X1,Y1,X2,Y2),ROI)
        print('write  ----IMAGE---- frame{}--- insectNum{} ---sex{}: {}_{}_{}_{} '.format(counter,insect_num,SEX,X1,Y1,X2,Y2), success)

    counter+=1
    success, image = video.read()

