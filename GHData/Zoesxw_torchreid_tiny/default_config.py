from yacs.config import CfgNode as CN


def get_default_config():
    cfg = CN()

    # model
    cfg.model = CN()
    cfg.model.name = 'resnet50'
    cfg.model.pretrained = True
    cfg.model.load_weights = ''  # evaluate
    cfg.model.resume = ''  # resume training

    # data
    cfg.data = CN()
    cfg.data.height = 256
    cfg.data.width = 128
    cfg.data.transforms = ['random_flip']
    cfg.data.norm_mean = [0.485, 0.456, 0.406]
    cfg.data.norm_std = [0.229, 0.224, 0.225]
    cfg.data.save_dir = 'log'
    
    # sampler
    cfg.sampler = CN()
    cfg.sampler.train_sampler = 'RandomSampler'
    cfg.sampler.num_instances = 4

    # train
    cfg.train = CN()
    cfg.train.optim = 'adam'
    cfg.train.lr = 0.0003
    cfg.train.weight_decay = 5e-4
    cfg.train.max_epoch = 60
    cfg.train.start_epoch = 0
    cfg.train.batch_size = 32
    cfg.train.fixbase_epoch = 0
    cfg.train.open_layers = ['classifier']
    cfg.train.staged_lr = False  # no
    cfg.train.new_layers = ['classifier']  # no
    cfg.train.base_lr_mult = 0.1  # no
    cfg.train.lr_scheduler = 'single_step'
    cfg.train.stepsize = [20]
    cfg.train.gamma = 0.1
    cfg.train.seed = 1

    # optimizer
    cfg.sgd = CN()
    cfg.sgd.momentum = 0.9
    cfg.sgd.dampening = 0.
    cfg.sgd.nesterov = False
    cfg.rmsprop = CN()
    cfg.rmsprop.alpha = 0.99
    cfg.adam = CN()
    cfg.adam.beta1 = 0.9
    cfg.adam.beta2 = 0.999

    # loss
    cfg.loss = CN()
    cfg.loss.softmax = CN()
    cfg.loss.softmax.label_smooth = True
    cfg.loss.triplet = CN()
    cfg.loss.triplet.margin = 0.3

    # test
    cfg.test = CN()
    cfg.test.batch_size = 100
    cfg.test.dist_metric = 'euclidean'
    cfg.test.normalize_feature = False
    cfg.test.evaluate = False  # evaluate
    cfg.test.eval_freq = 20
    cfg.test.rerank = False
    cfg.test.visrank = False

    return cfg


def imagedata_kwargs(cfg):
    return {
        'height': cfg.data.height,
        'width': cfg.data.width,
        'transforms': cfg.data.transforms,
        'norm_mean': cfg.data.norm_mean,
        'norm_std': cfg.data.norm_std,
        'batch_size_train': cfg.train.batch_size,
        'batch_size_test': cfg.test.batch_size,
        'num_instances': cfg.sampler.num_instances,
        'train_sampler': cfg.sampler.train_sampler,
    }


def optimizer_kwargs(cfg):
    return {
        'optim': cfg.train.optim,
        'lr': cfg.train.lr,
        'weight_decay': cfg.train.weight_decay,
        'momentum': cfg.sgd.momentum,
        'sgd_dampening': cfg.sgd.dampening,
        'sgd_nesterov': cfg.sgd.nesterov,
        'rmsprop_alpha': cfg.rmsprop.alpha,
        'adam_beta1': cfg.adam.beta1,
        'adam_beta2': cfg.adam.beta2,
        'staged_lr': cfg.train.staged_lr,
        'new_layers': cfg.train.new_layers,
        'base_lr_mult': cfg.train.base_lr_mult
    }


def lr_scheduler_kwargs(cfg):
    return {
        'lr_scheduler': cfg.train.lr_scheduler,
        'stepsize': cfg.train.stepsize,
        'gamma': cfg.train.gamma,
        'max_epoch': cfg.train.max_epoch
    }
