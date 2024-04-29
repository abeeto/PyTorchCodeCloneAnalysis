import os
import numpy as np
import torch
from torchreid.utils import FeatureExtractor


extractor = FeatureExtractor(
    model_name='osnet_x1_0',
    device='cpu')

os.chdir("./all_images/")
image_list = sorted(os.listdir())

features = extractor(image_list)
print(features.shape) # output (25, 512)
features_np = features.numpy()

os.chdir("../")
with open(r'feature_extraction.txt', 'w') as f:
    f.write(" ".join(map(str, features)))
    
## FAISS
import faiss
i = image_list.index(os.listdir("input_image/")[0])
distance_dict = {}
x = features_np[[i]]

for j in range(26):
    if i != j:
        q = features_np[[j]]
        index = faiss.index_factory(512, "Flat", faiss.METRIC_INNER_PRODUCT)
        index.ntotal
        faiss.normalize_L2(x)
        index.add(x)
        faiss.normalize_L2(q)
        distance, index = index.search(q, 5)
        distance_dict[j]= distance[0,0]
        distance_order = sorted(distance_dict.items(), key=lambda x: x[1], reverse=True)

print("ImageNo and Distance",distance_order[:7])
for k in range(7):
    print("sequentially",image_list[distance_order[:7][k][0]])

## SAVE IMAGES TO OUTPUT
import shutil
for k in range(7):
    image_list[distance_order[:7][k][0]]
    shutil.copy2('./all_images/'+image_list[distance_order[:7][k][0]], './output_images/') # complete target filename given
os.remove("./all_images/"+os.listdir("input_image/")[0])
