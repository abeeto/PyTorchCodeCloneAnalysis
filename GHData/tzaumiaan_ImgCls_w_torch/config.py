#lr, model_name = 0.3, 'shufflenet_v2_x1_0'
#lr, model_name = 0.1, 'mobilenet_v2'
lr, model_name = 0.01, 'inception_v3'
bs = 50
ckpt_name = '{}_sdd_bs{}_lr{}_ep{}'.format(model_name, bs, lr, 0)
