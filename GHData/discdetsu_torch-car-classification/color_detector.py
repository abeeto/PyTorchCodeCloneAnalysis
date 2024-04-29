import cv2
import pandas as pd

# Reading csv file with pandas and giving names to each column
index = ["color", "color_name", "hex", "R", "G", "B"]
csv = pd.read_csv('colors.csv', names=index, header=None)

# function to calculate minimum distance from all colors and get the most matching color
def get_color_name(pos, img):
    b, g, r = img[pos[1], pos[0]]
    b = int(b)
    g = int(g)
    r = int(r)
    minimum = 10000
    for i in range(len(csv)):
        d = abs(r - int(csv.loc[i, "R"])) + abs(g - int(csv.loc[i, "G"])) + abs(b - int(csv.loc[i, "B"]))
        if d <= minimum:
            minimum = d
            cname = csv.loc[i, "color_name"]
    return cname
