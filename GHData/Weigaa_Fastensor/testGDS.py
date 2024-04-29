import torch
from nvidia.dali import pipeline_def, fn
import nvidia.dali.plugin.pytorch
import os
import time
# misc python stuff
import numpy as np
from glob import glob
import shutil
import tempfile
import cupy
import kvikio
# visualization
from PIL import Image

# def plot_batch(np_arrays, nsamples=None):
#     if nsamples is None:
#         nsamples = len(np_arrays)
#     fig, axvec = plt.subplots(nrows=1, ncols=nsamples, figsize=(10, 10 * nsamples))
#     for i in range(nsamples):
#         ax = axvec[i]
#         ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
#         ax.imshow(Image.fromarray(np_arrays[i]))
#     plt.tight_layout()
device=torch.device('cuda:0')
batch_size = 1 # to be used in pipelines
dali_extra_dir = os.environ['DALI_EXTRA_PATH']
# data_dir_2d = os.path.join(dali_extra_dir, 'db', '3D', 'MRI', 'Knee', 'npy_2d_slices', 'STU00001')
# data_dir_3d = os.path.join(dali_extra_dir, 'db', '3D', 'MRI', 'Knee', 'npy_3d', 'STU00001')
# data_dir = os.path.join(data_dir_2d, 'SER00001')
# # Listing all *.npy files in data_dir
data_dir = '.'
files = sorted([f for f in os.listdir(data_dir) if '.npy' in f])
testmodelbsz = 64
# files  = files[0:5]

@pipeline_def(batch_size=batch_size, num_threads=8, device_id=0)
# def pipe_gds():
#     data = fn.readers.numpy(device='gpu', file_root=data_dir, files=files)
#     return data

#DALI_CPU_Batch
def pipe_gds():
    data = fn.readers.numpy(device='cpu', file_root=data_dir, files=files)
    return data

# def run(p):
#     p.build()  # build the pipeline
#     outputs = p.run()  # Run once
#     # Getting the batch as a list of numpy arrays, for displaying
#     batch = [np.array(outputs[0][s]) for s in range(batch_size)]
#     return batch

# print(files)
# def pipe_gds(filename):
#     data = fn.readers.numpy(device='gpu', file_root=data_dir, files=filename)
#     return data
N = 30
# average_save_tensor = 0
# average_save_numpy = 0
# average_save_cupy = 0
# #使用torch.save()写张量到SSD
# for i in range(N):
#     # Inputimages = torch.randn(testmodelbsz, 256, 56, 56).cuda()
#     Inputimages = torch.randn(testmodelbsz, 256, 56, 56)
#     # path = 'Inputtensor' + str(i) + '.pt'
#     path2 = 'Inputnumpy' + str(i) + '.npy'
#     # path3 = 'Inputcupy'+ str(i)
#     # Inputimages = torch.randn(512, 256, 56, 56)
#     # path = 'Inputtensor.pt'
#     # path2 = 'Inputnumpy.npy'
#     #save by torch.save()
#     # torch.cuda.synchronize()
#     # begin = time.time()
#     # torch.save(Inputimages, path)
#     # torch.cuda.synchronize()
#     # end = time.time()
#     # time1 = end -begin
#     # average_save_tensor += time1
#     # print("torchsave spendtime is", time1)
#     #save by GDS
#     # torch.cuda.synchronize()
#     # begin = time.time()
#     # Inputcupy = cupy.asarray(Inputimages)
#     # f = kvikio.CuFile(path3, "w")
#     # f.write(Inputcupy)
#     # f.close()
#     # torch.cuda.synchronize()
#     # end = time.time()
#     # time3 = end - begin
#     # print("cupysave spendtime is", time3)
#     # if i > 0:
#     #     average_save_cupy += time3
#     # save by numpy
#     torch.cuda.synchronize()
#     begin = time.time()
#     Inputnumpy = Inputimages.cpu().numpy()
#     # end = time.time()
#     # print("transfer time is", end - begin)
#     # torch.cuda.synchronize()
#     # begin = time.time()
#     np.save(path2, Inputnumpy)
#     torch.cuda.synchronize()
#     end = time.time()
#     time2 = end - begin
#     average_save_numpy += time2
#     print("numpysave spendtime is", time2)
#     # os.remove(path)
#     # os.remove(path2)
#     # os.remove(path3)
# print("average tensorsave spendtime is", average_save_tensor / N, "average numpysave spendtime is" , average_save_numpy / N,"average cupysave spendtime is" , average_save_cupy / (N -1))


# average_load_tensor = 0
# average_load_numpy = 0
# average_transfer_numpy = 0
# Inputimages = torch.empty(testmodelbsz, 256, 56, 56).to(device)
# for i in range(N):
#     path = 'Inputtensor' + str(i) + '.pt'
#     path2 = 'Inputnumpy' + str(i) + '.npy'
#     # # 使用torch.load()读到GPU的时间
#     # # path = 'Inputtensor.pt'
#     # torch.cuda.synchronize()
#     # begin = time.time()
#     # Inputimages = torch.load(path, map_location=lambda storage, loc: storage.cuda(0))
#     # torch.cuda.synchronize()
#     # end = time.time()
#     # time1 = end - begin
#     # average_load_tensor += time1
#     # print("torch.load spendtime is", time1)
#     #使用DALi读到GPU的时间
#     p = pipe_gds(filename=path2)
#     p.build()
#     torch.cuda.synchronize()
#     begin = time.time()
#     pipe_out = p.run()
#     torch.cuda.synchronize()
#     end = time.time()
#     time1 = end - begin
#     # print("numpyload spendtime is", time1)
#     # print(pipe_out[0][0])
#     torch.cuda.synchronize()
#     begin = time.time()
#     nvidia.dali.plugin.pytorch.feed_ndarray(pipe_out[0][0], Inputimages)
#     torch.cuda.synchronize()
#     end= time.time()
#     time2 = end - begin
#     # print("transfer time is", time2)
#     time3 = time1 + time2
#     if i > 1:
#         average_load_numpy += time3
#         average_transfer_numpy += time2
#     # print("load time", time1)
#     # print("transfer time", time2)
#     os.remove(path2)
#     print("total gdsload time",time3)
# print("average tensorload spendtime is", average_load_tensor / N , "average numpyload spendtime is" , average_load_numpy / (N - 2),"average transfer spendtime is" , average_transfer_numpy / (N - 2))

# #DALI_Batch Load
# average_transfer_pipeline_numpy = 0
# p = pipe_gds()
# p.build()
# for i in range(N):
#     torch.cuda.synchronize()
#     begin = time.time()
#     pipe_out= p.run()
#     torch.cuda.synchronize()
#     end = time.time()
#     time1 = end - begin
#     print("numpyload pipeline spendtime is", time1)
#     print(len(pipe_out[0]))
#     if i > 1:
#         average_transfer_pipeline_numpy += time1
# print("average numpyload pipeline spendtime is", average_transfer_pipeline_numpy / ((N -2) * batch_size))

#DALI_Batch Load_CPU
average_transfer_pipeline_numpy = 0
p = pipe_gds()
p.build()
for i in range(N):
    torch.cuda.synchronize()
    begin = time.time()
    pipe_out= p.run()
    torch.cuda.synchronize()
    end = time.time()
    time1 = end - begin
    print("numpyload pipeline spendtime is", time1)
    print(len(pipe_out[0]))
    if i > 1:
        average_transfer_pipeline_numpy += time1
print("average numpyload pipeline spendtime is", average_transfer_pipeline_numpy / ((N -2) * batch_size))

# averagemovetime = 0
# #CPU到GPU传输测试
# for i in range(N):
#     Inputimages = torch.zeros(testmodelbsz, 256, 56, 56)
#     torch.cuda.synchronize()
#     begin = time.time()
#     Inputimages.to(device)
#     torch.cuda.synchronize()
#     end = time.time()
#     if i > 0:
#         averagemovetime += end - begin
#     print("move time is", end - begin)
# print("average move time is", averagemovetime/(N -1 ))

# #使用Numpy读数据时间
# average_load_numpy = 0
# for i in range(N):
#     path = 'Inputnumpy' + str(i) + '.npy'
#     # 使用np.load()读到GPU的时间
#     torch.cuda.synchronize()
#     begin = time.time()
#     Inputimages = torch.from_numpy(np.load(path)).to(device)
#     torch.cuda.synchronize()
#     end = time.time()
#     time1 = end - begin
#     if i > 1:
#         average_load_numpy += time1
#     print("torch.load spendtime is", time1)
#     os.remove(path)
# print("average numpyload spendtime is" , average_load_numpy / (N - 2))

# # #使用kvikio读数据时间
# averageloadtime = 0
# #CPU到GPU传输测试
# cupyimages = cupy.asarray(torch.empty(testmodelbsz, 256, 56, 56))
# for i in range(N):
#     torch.cuda.synchronize()
#     path3 = 'Inputcupy' + str(i)
#     begin = time.time()
#     Inputimages = cupy.empty_like(cupyimages)
#     f = kvikio.CuFile(path3, "r")
#     # Read whole array from file
#     f.read(Inputimages)
#     Inputtensor = torch.as_tensor(Inputimages, device = device)
#     torch.cuda.synchronize()
#     end = time.time()
#     if i > 0:
#         averageloadtime += end - begin
#     print("load time is", end - begin)
#     os.remove(path3)
# print("average load time is", averageloadtime/(N -1))



# data_gds = pipe_out[0].as_cpu().as_array()  # as_cpu() to copy the data back to CPU memory
# print(data_gds.shape)
