import numpy as np

bound=[100,100,5]
max_vales = [bound[0], bound[1], bound[2], 2, 2, 2, bound[0], bound[1], bound[2], 1,1, 1, 4, 4, 4, 1, 1, 1, bound[0], bound[1], bound[2]]
min_vales = [0, 0, 0, -2, -2, -2, 0, 0, 0, -1, -1, -1, 4, 4, 4, 1, 1, 1, -bound[0], -bound[1], -bound[2]]
max_vales = np.array(max_vales)
min_vales = np.array(min_vales)
all = np.stack((max_vales, min_vales),axis=1)
ptp_scale = np.ptp(all,axis=1)

print(ptp_scale)

a = [33,33,2,-1,0,1,33,33,-1,0,1,1,1,1,1,1,1,-33,0,0]

d = 2.*(a - np.min(a))/np.ptp(a)-1