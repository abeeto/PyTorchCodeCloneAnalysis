import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
#%% Read a picture
img = cv.imread('/home/pengfei/projects/PyTorchStudy/heaterpattern.png',0)
ret,img_bin = cv.threshold(img,127,255,cv.THRESH_BINARY)
plt.imshow(img_bin)
df_tosave = pd.DataFrame(img_bin)
df_tosave.to_csv("temp_df_save.csv")
# coordinate of the first dark, or the starting point of the maze
start_pix_rowid = 537
start_pix_colid = 1248
start_pix_color = img_bin[start_pix_rowid,start_pix_colid]
img_bin_copy = copy.deepcopy(img_bin)
img_bin_copy[start_pix_rowid,start_pix_colid] = 120
plt.imshow(img_bin_copy)

#%% using breadth first search
visited = [] # List to keep track of visited nodes.
queue = []     #Initialize a queue
counter = 0
image_count = 0

node = [start_pix_rowid, start_pix_colid]
visited.append(node)
queue.append(node)

while queue:
    s = queue.pop(0) 
    #print (s, end = " ")
    # img_bin_copy[s[0],s[1]] = 120
    # plt.imshow(img_bin_copy)
    # plt.pause(0.001)
    # plt.show()
    counter = counter + 1
    if counter > 10000:
        image_count = image_count + 1
        for content in visited:
            img_bin_copy[content[0],content[1]] = 120
        plt.imshow(img_bin_copy)
        plt.savefig("image_{}".format(image_count))
        plt.close("all")
        #plt.pause(0.001)
        #plt.show()
        print("draw picture")
        counter = 0
        visited = []
        #queue = []



    for idx in range(4):
      # pixright_rowid = s[0]
      # pixright_colid = s[1] + 1
      pixneighbor_ids = [[s[0],s[1]+1],[s[0]-1,s[1]],[s[0],s[1]-1],[s[0]+1,s[1]]]
      pixneighbor_ids_real = []
      for pix_id in pixneighbor_ids:
        if pix_id[0] < 1091 and pix_id[1] < 1342 and img_bin_copy[pix_id[0],pix_id[1]] < 1:
          pixneighbor_ids_real.append(pix_id)

    for neighbor in pixneighbor_ids_real:
        if neighbor not in visited:
            visited.append(neighbor)
            queue.append(neighbor)