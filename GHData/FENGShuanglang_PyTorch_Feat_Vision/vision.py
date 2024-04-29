import numpy as np
import matplotlib.pyplot as plt
import os
npyfile=np.load(r"G:\KeTi\JBHI_paper\图\skin lesion 可视化\f5\ISIC_0000003_segmentation.npy")
feature_path=r'G:\KeTi\JBHI_paper\图\skin lesion 可视化\feature'
npy=np.squeeze(npyfile)

ave=np.mean(npy,axis=0)
ave1=np.reshape(ave,(1,)+ave.shape)

con=np.concatenate((npy,ave1),axis=0)

feat_data=con
for i in range(len(feat_data)):
    fig = plt.figure(figsize=(10,10))#
    im = plt.imshow(feat_data[i], cmap=plt.get_cmap('jet'),interpolation='nearest')#, # important function
    plt.colorbar(im,shrink=0.4)
    s=os.path.join(feature_path,str(i)+'_reponse_map.png')
    fig.savefig(s, dpi=100,bbox_inches='tight')# 
plt.show()
