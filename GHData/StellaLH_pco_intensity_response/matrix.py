# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 09:54:07 2017

@author: stella
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.optimize import curve_fit

## exposure times (better to choose manually)
# exp_time=[0.1,0.2,0.5]

## intensities

#datasheet_ints=[0.35,0.098,0.0343,0.007,0.00245,0.000784,0.0002401,0.000098] #for wavelengths 400-600
datasheet_ints=[0.35,0.098,0.0343,0.007,0.00245,0.000784,0.0002401] #for wavelengths 400-600

filter_value=[]
count=[]

exp_time='0p1'
final_array=np.empty((3,2048,2060))

""" Open each file, find average intensity over the 100 frames """
fid=h5py.File('/media/stella/F0B427B9B42780E8/Stella/exp_time0p1/intensity_0p320000/pco_000001.h5', 'r')
data=fid['/entry/data']
mean0p320000=np.mean(data, axis=0)
print 'Average intensity at T = 0.32000 found'

fid=h5py.File('/media/stella/F0B427B9B42780E8/Stella/exp_time0p1/intensity_0p100000/pco_000001.h5', 'r')
data=fid['/entry/data']
mean0p100000=np.mean(data, axis=0)
print 'Average intensity at T = 0.10000 found' 

fid=h5py.File('/media/stella/F0B427B9B42780E8/Stella/exp_time0p1/intensity_0p032000/pco_000001.h5', 'r')
data=fid['/entry/data']
mean0p032000=np.mean(data, axis=0)
print 'Average intensity at T = 0.03200 found'

fid=h5py.File('/media/stella/F0B427B9B42780E8/Stella/exp_time0p1/intensity_0p010000/pco_000001.h5', 'r')
data=fid['/entry/data']
mean0p010000=np.mean(data, axis=0)
print 'Average intensity at T = 0.01000 found'

fid=h5py.File('/media/stella/F0B427B9B42780E8/Stella/exp_time0p1/intensity_0p003200/pco_000001.h5', 'r')
data=fid['/entry/data']
mean0p003200=np.mean(data, axis=0)
print 'Average intensity at T = 0.00320 found'

fid=h5py.File('/media/stella/F0B427B9B42780E8/Stella/exp_time0p1/intensity_0p001000/pco_000001.h5', 'r')
data=fid['/entry/data']
mean0p001000=np.mean(data, axis=0)
print 'Average intensity at T = 0.00100 found'

fid=h5py.File('/media/stella/F0B427B9B42780E8/Stella/exp_time0p1/intensity_0p000320/pco_000001.h5', 'r')
data=fid['/entry/data']
mean0p000320=np.mean(data, axis=0)
print 'Average intensity at T = 0.00032 found'

""" Analyse each pixel"""
count=[]
for ypixel in range(2048):
    for xpixel in range(2060):
        #append average pixel value to count
        print 'pixel x=', xpixel, 'y=' ,ypixel
        count.append(mean0p320000[ypixel, xpixel])
        #print 'data 1 added to count'
        count.append(mean0p100000[ypixel, xpixel])
        #print 'data 2 added to count'
        count.append(mean0p032000[ypixel, xpixel])
        #print 'data 3 added to count'
        count.append(mean0p010000[ypixel, xpixel])
        #print 'data 4 added to count'
        count.append(mean0p003200[ypixel, xpixel])
        #print 'data 5 added to count'
        count.append(mean0p001000[ypixel, xpixel])
        #print 'data 6 added to count'
        count.append(mean0p000320[ypixel, xpixel])
        #print 'data 7 added to count'
     
        # fit count against data sheet filter values
        fit= np.polyfit(datasheet_ints, count,1)
        line=np.poly1d(fit)
        corr=np.corrcoef(line(datasheet_ints), count)[0,1]
        # add fit values to final array
        final_array[0,ypixel,xpixel]=fit[1] #intercept
        final_array[1,ypixel,xpixel]=fit[0] #gradient
        final_array[2,ypixel,xpixel]=straight_coeff #correlation coefficient
        count=[]
        

