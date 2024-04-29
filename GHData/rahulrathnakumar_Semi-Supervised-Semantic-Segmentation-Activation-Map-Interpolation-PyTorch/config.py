configDict = {
    'directory_name': 'DEBUG_3',
    'root_dir' : 'data/NEU/',
    'num_classes' : 2,
    'labeled_batch_size' : 5,
    'unlabeled_batch_size': 10,
    'weakly_labeled_batch_size': 10,
    'epochs' : 600,
    'lr' : 1e-2,
    'momentum' : 0.9,
    'optim_w_decay' : 1e-5,
    'step_size' : 300,
    'gamma' : 0.1,
    'ema_decay': 0.99,
    'alpha_ict': 0.5,
    'num_labeled': 65, 
    'num_unlabeled': 70,
    'consistency_weight': 'none', # Options: 'none', 'ramp', 'dynamic' 
    'load_ckp': False,
    'print_gpu_usage': False
}
