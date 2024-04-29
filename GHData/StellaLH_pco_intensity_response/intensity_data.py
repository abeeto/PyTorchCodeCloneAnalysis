# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 11:53:51 2017

@author: stella
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.optimize import curve_fit

#intensities=['320000','100000','032000','010000','003200','001000','000320','000100']
intensities=['100000','032000','010000','003200','001000','000320','000100']
#datasheet_ints=[0.39,0.1,0.039,0.008,0.00312,0.0008,0.000312,0.00008]
datasheet_ints=[0.35,0.098,0.0343,0.007,0.00245,0.000784,0.0002401,0.000098] #for wavelengths 400-600
other_filters=['sharif']

filter_value=[]
count=[]
adjfilter=[]

filter_value2=[]
count2=[]
adjfilter2=[]

# lab filters
for i in range(len(intensities)):
    fid=h5py.File('/media/stella/F0B427B9B42780E8/Stella/exp_time0p2/intensity_0p%s/pco_000001.h5' %intensities[i],'r')
    data=fid['/entry/data']
    shape=data.shape
    #print 'intensity =',i
    for j in range(shape[0]):
        print 'frame =' ,j
        for pixelx in range(1290,1310):
            print 'x=',pixelx
            for pixely in range(1290,1310):
                print 'y=' ,pixely
                filter_value.append(float('0.%s' %intensities[i]))
                adjfilter.append(datasheet_ints[i])
                
                count.append(data[j,pixely, pixelx])
    fid.close()
# sharif's filters
sharif_filters=intensities[:-1]
for k in range(len(sharif_filters)):
    fid=h5py.File('/media/stella/F0B427B9B42780E8/Stella/exp_time0p2/sharif/intensity_0p%s/pco_000001.h5' %intensities[k],'r')
    data=fid['/entry/data']
    shape=data.shape
    for l in range(shape[0]):
        print 'frame =' ,l
        for pixelx in range(1290,1310):
            print 'x=',pixelx
            for pixely in range(1290,1310):
                print 'y=' ,pixely
                filter_value2.append(float('0.%s' %intensities[k])/4.0)
                adjfilter2.append(datasheet_ints[k]*0.2)
                
                count2.append(data[l,pixely, pixelx])    
# dark frame   
fid=h5py.File('/media/stella/F0B427B9B42780E8/Stella/exp_time0p2/dark/pco_000001.h5','r')
data=fid['/entry/data']
shape=data.shape
for m in range(shape[0]):
    for pixelx in range(1290,1310):
        print 'x=',pixelx
        for pixely in range(1290,1310):
            print 'y=' ,pixely
            filter_value.append(0.0)
            adjfilter.append(0.0)
            count.append(data[m,pixely, pixelx])

fitx=np.arange(0,1, 0.001)   
count3=np.concatenate((count, count2))             

##fits for average filter values
filter_value3=np.concatenate((filter_value, filter_value2))
#-------------------------------------------
line=np.polyfit(filter_value, count,1)
line2=np.polyfit(filter_value2,count2, 1)
line3=np.polyfit(filter_value3, count3,1)
#-------------------------------------------
fit=fitx*line[0]+line[1]
fit2=fitx*line2[0]+line2[1]
fit3=fitx*line3[0]+line3[1]

##fits for exact filter values on data sheets
adjfilter3=np.concatenate((adjfilter, adjfilter2))
#-------------------------------------------
adjline=np.polyfit(adjfilter, count, 1)
adjline2=np.polyfit(adjfilter2, count2, 1)
adjline3=np.polyfit(adjfilter3, count3,1)
#-------------------------------------------
adjfit=fitx*adjline[0]+adjline[1]
adjfit2=fitx*adjline2[0]+adjline2[1]
adjfit3=fitx*adjline3[0]+adjline3[1]

#fig, ax = plt.subplots(nrows=1,ncols=2)
plt.figure(figsize=(10,5))  
plt.subplots_adjust(wspace=0.5)


plt.subplot(1,2,1)
plt.scatter(filter_value, count, label='Lab filters', color='c')
plt.plot(fitx, fit, label='Lab filters fit', color='c')
plt.scatter(filter_value2, count2, label='Sharifs filters', color='m')
plt.plot(fitx, fit2, label='Sharifs filters fit', color='m')
plt.plot(fitx, fit3, label='Concatenated fit', color='g')
plt.title('Average Filter Values')
plt.xlabel('Filter Tansmission')
plt.ylabel('Counts')
plt.xlim(0,1)
plt.legend(loc=2, fontsize='xx-small')


plt.subplot(1,2,2)
plt.scatter(adjfilter, count, label='Lab filters', color='c')
plt.scatter(adjfilter2, count2, label='Sharifs filters', color='m')
plt.plot(fitx, adjfit,  label='Lab filters fit', color='c')
plt.plot(fitx, adjfit2,  label='Sharifs filters fit', color='m')
plt.plot(fitx, adjfit3, label='Concatenated fit', color='g')
plt.title('Data Sheet Filter Values\n for Wavelength Used')
plt.xlabel('Filter Tansmission')
plt.ylabel('Counts')
plt.xlim(0,1)
plt.legend(loc=2, fontsize='xx-small')



              