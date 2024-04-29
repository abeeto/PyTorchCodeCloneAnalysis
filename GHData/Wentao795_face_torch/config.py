import os
class Defaultconfig(object):
    train_path = ''
    train_data = './dataset/'+train_path.split('/')[-1]
    embedding_size = 128
    loss_type = 0 # 0 softmaxe 1 arcface 3 am_softmax
    num_classe = 180000
    margin_s = 64
    margin_m = 0.5
    gpu_id = [0,1,2,3]
    lr = 0.1
    momentum = 0.9
    weight_decay = 5e-4
    batch_size = 512
    num_work = 128
    resume = 0
    model_path = ''
    model_output = ''
    end_epoch = 100
    model_name = 'face_mobile'

config = Defaultconfig()