"""
author: Yasamin Jafarian
"""

import tensorflow as tf
import os.path
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import skimage.data
from PIL import Image, ImageDraw, ImageFont
import random
import sys
import matplotlib.pyplot as plt
tf.logging.set_verbosity(tf.logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import pdb
import math
from tensorflow.python.platform import gfile
import scipy.misc
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from hourglass_net_depth import hourglass_refinement
from hourglass_net_normal import hourglass_normal_prediction
from hourglass_net_depth_torch import hourglass_refinement_1
from hourglass_net_normal_torch import hourglass_normal_prediction_1
from utils import (write_matrix_txt,get_origin_scaling,get_concat_h, depth2mesh, read_test_data, nmap_normalization, get_test_data) 
from training.training_code.utils.Loss_functions_torch import calc_loss_normal2_1, calc_loss_1, calc_loss_d_refined_mask_1
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "5"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
############################## test path and outpath ##################################
def random_List(size):
    result = []
    for v in range(size):
        result.append(random.randint(1, 339))
    return result
num_sub=5
result=random_List(num_sub)
print(result)
img_number_origin="/{0:05d}/"

#data_main_path = './test_data'+img_number
#outpath = './test_data'+"/infer_out/"+"tensorflow" +img_number
#visualization = True
num=3
##############################    Inference Code     ##################################
pre_ck_pnts_dir_DR_tensor =  "/home/ug_psh/HDNet_torch_ami/training_progress/tensorflow/model/HDNet/"
pre_ck_pnts_dir_DR = "/home/ug_psh/HDNet_torch_ami/model/depth_prediction/"
#pre_ck_pnts_dir_DR_torch =  "/home/ug_psh/HDNet_torch_ami/training_progress/pytorch/model/HDNet_up5/"
model_num_DR = '1920000'
pre_ck_pnts_dir_NP_tensor =  "/home/ug_psh/HDNet_torch_ami/training_progress/tensorflow/model/NormalEstimator/"
pre_ck_pnts_dir_NP = "/home/ug_psh/HDNet_torch_ami/model/normal_prediction/"
#pre_ck_pnts_dir_NP_torch =  "/home/ug_psh/HDNet_torch_ami/training_progress/pytorch/model/HDNet_up5/"
model_num_NP = '1710000'
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

# Creat the outpath if not exists
# Vis_dir = outpath
# if not gfile.Exists(Vis_dir):
#     print("Vis_dir created!")
#     gfile.MakeDirs(Vis_dir)
refineNet_graph = tf.Graph()
NormNet_graph = tf.Graph()

# Define the depth and normal networks
# ***********************************Normal Prediction******************************************
with NormNet_graph.as_default():
    x1_n = tf.placeholder(tf.float32, shape=(None, 256,256,3))
    with tf.variable_scope('hourglass_normal_prediction', reuse=tf.AUTO_REUSE):
        out2_normal = hourglass_normal_prediction(x1_n,True)

print("Model_torch NP restored.")
# ***********************************Depth Prediction******************************************
with refineNet_graph.as_default():
    x1 = tf.placeholder(tf.float32, shape=(None, 256,256,9))
    with tf.variable_scope('hourglass_stack_fused_depth_prediction', reuse=tf.AUTO_REUSE):
        out2_1 = hourglass_refinement(x1,True)

print("Model_torch DR restored.")
# load checkpoints
sess4 = tf.Session(graph=NormNet_graph)
sess3 = tf.Session(graph=refineNet_graph)
sess2 = tf.Session(graph=NormNet_graph)
sess = tf.Session(graph=refineNet_graph)
with sess.as_default():
    with refineNet_graph.as_default():
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
        saver = tf.train.import_meta_graph(pre_ck_pnts_dir_DR+'model_1920000.ckpt.meta')
        saver.restore(sess,pre_ck_pnts_dir_DR+'model_1920000.ckpt')
        print("Model_tensor DR restored.")
with sess2.as_default():
    with NormNet_graph.as_default():
        tf.global_variables_initializer().run()
        saver2 = tf.train.Saver()
        saver2 = tf.train.import_meta_graph(pre_ck_pnts_dir_NP+'model_1710000.ckpt.meta')
        saver2.restore(sess2,pre_ck_pnts_dir_NP+'model_1710000.ckpt')
        print("Model_tensor NP restored.")
with sess3.as_default():
    with refineNet_graph.as_default():
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
        saver = tf.train.import_meta_graph(pre_ck_pnts_dir_DR_tensor+'model_11398.ckpt.meta')
        saver.restore(sess,pre_ck_pnts_dir_DR_tensor+'model_11398.ckpt')
        print("Model_tensor DR restored.")
with sess4.as_default():
    with NormNet_graph.as_default():
        tf.global_variables_initializer().run()
        saver2 = tf.train.Saver()
        saver2 = tf.train.import_meta_graph(pre_ck_pnts_dir_NP_tensor+'model_6000.ckpt.meta')
        saver2.restore(sess2,pre_ck_pnts_dir_NP_tensor+'model_6000.ckpt')
        print("Model_tensor NP restored.")
    # Read the test images and run the HDNet
N_error_total=[]
D_error_total=[]
R_error_total=[]
N_error_total_std=[]
D_error_total_std=[]
R_error_total_std=[]
N_error=[]
D_error=[]
R_error=[]
for i in range(num_sub+1):
    img_number=img_number_origin.format(result[i])
    data_main_path = '/local_data/TikTok_dataset'+img_number
    test_files = get_test_data(data_main_path,num)
    for f in range(len(test_files)):
        data_name = test_files[f]
        print('Processing file: ',data_name)
        X,Z, Z3, _, _,_,_, _,_,  _, _, DP = read_test_data(data_main_path,data_name,IMAGE_HEIGHT,IMAGE_WIDTH)
        
        prediction1n = sess2.run([out2_normal],feed_dict={x1_n:X})
        prediction1n_torch = model_n(torch.Tensor(X).type(torch.float32))
        normal_pred_raw_1  = torch.Tensor(np.asarray(prediction1n)[0,...]).type(torch.float32)
        normal_pred_raw_torch_1  = torch.Tensor(np.asarray(prediction1n_torch.detach().numpy())).type(torch.float32)
        normal_pred_raw  = np.asarray(prediction1n)[0,...]
        normal_pred_raw_torch  = np.asarray(prediction1n_torch.detach().numpy())
        normal_pred_raw_1  = torch.Tensor(np.asarray(prediction1n)[0,...]).type(torch.float32).to(device)
        normal_pred_raw_torch_1  = torch.Tensor(np.asarray(prediction1n_torch.detach().numpy())).type(torch.float32).to(device)
        Z_1=torch.Tensor(Z).type(torch.bool).to(device)
        N_error.append(calc_loss_normal2_1(normal_pred_raw_torch_1,normal_pred_raw_1,Z_1).detach().cpu().numpy())
        #pdb.set_trace()
        normal_pred = nmap_normalization(normal_pred_raw)
        normal_pred_torch = nmap_normalization(normal_pred_raw_torch)
        normal_pred = np.where(Z3,normal_pred,np.zeros_like(normal_pred))
        normal_pred_torch = np.where(Z3,normal_pred_torch,np.zeros_like(normal_pred_torch))
        X_1 = np.zeros((1,IMAGE_HEIGHT,IMAGE_WIDTH,9),dtype='f') 
        X_1_torch = np.zeros((1,IMAGE_HEIGHT,IMAGE_WIDTH,9),dtype='f')
        X_1[...,0]=X[...,0]
        X_1[...,1]=X[...,1]
        X_1[...,2]=X[...,2]
        X_1[...,3]=normal_pred[...,0]
        X_1[...,4]=normal_pred[...,1]
        X_1[...,5]=normal_pred[...,2]
        X_1[...,6]=DP[...,0]
        X_1[...,7]=DP[...,1]
        X_1[...,8]=DP[...,2]
        X_1_torch[...,0]=X[...,0]
        X_1_torch[...,1]=X[...,1]
        X_1_torch[...,2]=X[...,2]
        X_1_torch[...,3]=normal_pred_torch[...,0]
        X_1_torch[...,4]=normal_pred_torch[...,1]
        X_1_torch[...,5]=normal_pred_torch[...,2]
        X_1_torch[...,6]=DP[...,0]
        X_1_torch[...,7]=DP[...,1]
        X_1_torch[...,8]=DP[...,2]
        prediction1 = sess.run([out2_1],feed_dict={x1:X_1})
        prediction1_torch = model_d(torch.Tensor(X_1_torch).type(torch.float32))
        image  = np.asarray(prediction1)[0,0,...]
        image_torch  = np.asarray(prediction1_torch.detach().numpy())[0,...]
        image_1=torch.Tensor(image).type(torch.float32).to(device)
        image_torch_1=torch.Tensor(image_torch).type(torch.float32).to(device)
        D_error.append(calc_loss_1(image_torch_1,image_1,Z_1).detach().cpu().numpy())
        R_error.append(calc_loss_d_refined_mask_1(image_torch_1,image_1,Z_1,device).detach().cpu().numpy())
    N_error_total.append(np.mean(N_error))
    N_error_total_std.append(np.std(N_error))
    N_error.clear()
    D_error_total.append(np.mean(D_error))
    D_error_total_std.append(np.std(D_error))
    D_error.clear()
    R_error_total.append(np.mean(R_error))
    R_error_total_std.append(np.std(R_error))
    R_error.clear()
    print("N_error: %.4f +- %.4f, D_error: %.4f +- %.4f, R_error: %.4f +- %.4f" % (np.mean(N_error_total),np.mean(N_error_total_std),np.mean(D_error_total),np.mean(D_error_total_std),np.mean(R_error_total),np.mean(R_error_total_std)))
print("final N_error: %.4f +- %.4f, D_error: %.4f +- %.4f, R_error: %.4f +- %.4f" % (np.mean(N_error_total),np.mean(N_error_total_std),np.mean(D_error_total),np.mean(D_error_total_std),np.mean(R_error_total),np.mean(R_error_total_std)))
        # imagen = normal_pred[0,...]
        # #pdb.set_trace()    
        # write_matrix_txt(image*Z[0,...,0],Vis_dir+data_name+".txt")
        # write_matrix_txt(imagen[...,0]*Z[0,...,0],Vis_dir+data_name+"_normal_1.txt")
        # write_matrix_txt(imagen[...,1]*Z[0,...,0],Vis_dir+data_name+"_normal_2.txt")
        # write_matrix_txt(imagen[...,2]*Z[0,...,0],Vis_dir+data_name+"_normal_3.txt")
        # depth2mesh(image*Z[0,...,0], Z[0,...,0], Vis_dir+data_name+"_mesh")
        # if visualization:
        #     depth_map = image*Z[0,...,0]
        #     normal_map = imagen*Z3[0,...]
        #     min_depth = np.amin(depth_map[depth_map>0])
        #     max_depth = np.amax(depth_map[depth_map>0])
        #     depth_map[depth_map < min_depth] = min_depth
        #     depth_map[depth_map > max_depth] = max_depth
            
        #     normal_map_rgb = -1*normal_map
        #     normal_map_rgb[...,2] = -1*((normal_map[...,2]*2)+1)
        #     normal_map_rgb = np.reshape(normal_map_rgb, [256,256,3]);
        #     normal_map_rgb = (((normal_map_rgb + 1) / 2) * 255).astype(np.uint8);
            
        #     plt.imsave(Vis_dir+data_name+"_depth.png", depth_map, cmap="hot") 
        #     plt.imsave(Vis_dir+data_name+"_normal.png", normal_map_rgb) 
        #     #pdb.set_trace()
        #     d = np.array(scipy.misc.imread(Vis_dir+data_name+"_depth.png"),dtype='f')
        #     d = np.where(Z3[0,...]>0,d[...,0:3],255.0)
        #     n = np.array(scipy.misc.imread(Vis_dir+data_name+"_normal.png"),dtype='f')
        #     n = np.where(Z3[0,...]>0,n[...,0:3],255.0)
        #     final_im = get_concat_h(Image.fromarray(np.uint8(X[0,...])),Image.fromarray(np.uint8(d)))
        #     final_im = get_concat_h(final_im,Image.fromarray(np.uint8(n)))
        #     final_im.save(Vis_dir+data_name+"_results.png")
            
        #     os.remove(Vis_dir+data_name+"_depth.png")
        #     os.remove(Vis_dir+data_name+"_normal.png")
    
