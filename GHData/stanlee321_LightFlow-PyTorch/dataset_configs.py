"""
Add dataset configurations here. Each dataset must have the following structure:
NAME = {
    IMAGE_HEIGHT: int,
    IMAGE_WIDTH: int,
    ITEMS_TO_DESCRIPTIONS: {
        'image_a': 'A 3-channel image.',
        'image_b': 'A 3-channel image.',
        'flow': 'A 2-channel optical flow field',
    },
    SIZES: {
        'train': int,
        'validate': int,    (optional)
        ...
    },
    BATCH_SIZE: int,
    PATHS: {
        'train': '',
        'validate': '', (optional)
        ...
    }
}
"""

"""
note that one step = one batch of data processed, ~not~ an entire epoch
'coeff_schedule_param': {
    'half_life': 50000,         after this many steps, the value will be i + (f - i)/2
    'initial_coeff': 0.5,       initial value
    'final_coeff': 1,           final value
},
"""




TRAINING_CONFIGS = {
    'START_EPOCH': 1,
    'TOTAL_EPOCHS': 10000,
    'BATCH_SIZE': 8,
    'TRAIN_N_BATCHES': -1,          # 'Number of min-batches per epoch. If < 0, it will be determined by training_dataloader'
    'CROP_SIZE': [384,512],         # [256, 256], #"Spatial dimension to crop training samples for training"
    'SCHEDULE_LR_FRECUENCY': 0,     # in number of iterations (0 for no schedule)'
    'SCHEDULE_LR_FRACTION': 10,
    'RGB_MAX': 255.,
    'NUMBER_WORKERS': 4,
    'NUMBER_GPUS': 1,
    'NO_CUDA': False,
    'SEED': 1,
    'NAME': 'run',                  # a name to append to the save directory
    'SAVE': './work',
    'VALIDATION_FRECUENCY': 5,      # Validate every n epochs
    'RENDER_VALIDATION': False,     # 'run inference (save flows to file) and every validation_frequency epoch'
    'INFERENCE': False,
    'INFERENCE_SIZE': [-1, -1],     #'spatial size divisible by 64. default (-1,-1) - largest possible valid size would be used'
    'INFERENCE_BATCH_SIZE': 1,
    'INFERENCE_N_BATCHES': -1,
    'SAVE_FLOW': False,              # Save predicted flows to file
    'RESUME': './work/LightFlow_train-checkpoint.pth.tar', # path to latest checkpoint (default: none)
    'LOG_FRECUENCY': 10,
    'SUMM_ITER': 10,                 # Log every n batches
    'SKIP_TRANING': False,           # 
    'SKIP_VALIDATION': False,
    'FP16': False,                   #Run model in pseudo-fp16 mode (fp16 storage fp32 math).'
    'FP16_SCALE': 1024.,             #Loss scaling, positive power of 2 values can improve fp16 convergence.
}

MODEL_CONFIGS = {
    'MODEL': 'LightFlow',
    'LOSS': 'L1Loss',               #L2Loss
    'OPTIMIZER': {
        'default': 'Adam',
        'skip_params': 'params',
        'parameter_defaults': {'weight_decay': 0.00004}
    },
    'TRAINING_DATASET': {
        'default': 'MpiSintelFinal',
        'skip_params': ['is_cropped'],
        'parameter_defaults': {'root': './MPI-Sintel/flow/training'}
    },
    'VALIDATION_DATASET': {
        'default': 'MpiSintelClean',
        'skip_params': ['is_cropped'],
        'parameter_defaults': {
            'root': './MPI-Sintel/flow/training',
            'replicates':1
        }
    },
    'INFERENCE_DATASET': {
        'default': 'MpiSintelClean',
        'skip_params': ['is_cropped'],
        'parameter_defaults': {
            'root': './MPI-Sintel/flow/training',
            'replicates':1
        }
    }
                

}



DATASET_CONFIG = {
    'IMAGE_HEIGHT': 384,
    'IMAGE_WIDTH': 512,
    'ITEMS_TO_DESCRIPTIONS': {
        'image_a': 'A 3-channel image.',
        'image_b': 'A 3-channel image.',
        'flow': 'A 2-channel optical flow field',
    },
    'SIZES': {
        'train': 22232,
        'validate': 640,
        'sample': 8,
    },
}


FLYING_CHAIRS_DATASET_CONFIG = {
    'IMAGE_HEIGHT': 384,
    'IMAGE_WIDTH': 512,
    'ITEMS_TO_DESCRIPTIONS': {
        'image_a': 'A 3-channel image.',
        'image_b': 'A 3-channel image.',
        'flow': 'A 2-channel optical flow field',
    },
    'SIZES': {
        'train': 22232,
        'validate': 640,
        'sample': 8,
    },
    'BATCH_SIZE': 8,
    'PATHS': {
        'train': './data/tfrecords/fc_train.tfrecords',
        'validate': './data/tfrecords/fc_val.tfrecords',
        'sample': './data/tfrecords/fc_sample.tfrecords',
    },
    'PREPROCESS': {
        'scale': False,
        'crop_height': 320,
        'crop_width': 448,
        'image_a': {
            'translate': {
                'rand_type': "uniform_bernoulli",
                'exp': False,
                'mean': 0,
                'spread': 0.4,
                'prob': 1.0,
            },
            'rotate': {
                'rand_type': "uniform_bernoulli",
                'exp': False,
                'mean': 0,
                'spread': 0.4,
                'prob': 1.0,
            },
            'zoom': {
                'rand_type': "uniform_bernoulli",
                'exp': True,
                'mean': 0.2,
                'spread': 0.4,
                'prob': 1.0,
            },
            'squeeze': {
                'rand_type': "uniform_bernoulli",
                'exp': True,
                'mean': 0,
                'spread': 0.3,
                'prob': 1.0,
            },
            'noise': {
                'rand_type': "uniform_bernoulli",
                'exp': False,
                'mean': 0.03,
                'spread': 0.03,
                'prob': 1.0,
            },
        },
        # All preprocessing to image A will be applied to image B in addition to the following.
        'image_b': {
            'translate': {
                'rand_type': "gaussian_bernoulli",
                'exp': False,
                'mean': 0,
                'spread': 0.03,
                'prob': 1.0,
            },
            'rotate': {
                'rand_type': "gaussian_bernoulli",
                'exp': False,
                'mean': 0,
                'spread': 0.03,
                'prob': 1.0,
            },
            'zoom': {
                'rand_type': "gaussian_bernoulli",
                'exp': True,
                'mean': 0,
                'spread': 0.03,
                'prob': 1.0,
            },
            'gamma': {
                'rand_type': "gaussian_bernoulli",
                'exp': True,
                'mean': 0,
                'spread': 0.02,
                'prob': 1.0,
            },
            'brightness': {
                'rand_type': "gaussian_bernoulli",
                'exp': False,
                'mean': 0,
                'spread': 0.02,
                'prob': 1.0,
            },
            'contrast': {
                'rand_type': "gaussian_bernoulli",
                'exp': True,
                'mean': 0,
                'spread': 0.02,
                'prob': 1.0,
            },
            'color': {
                'rand_type': "gaussian_bernoulli",
                'exp': True,
                'mean': 0,
                'spread': 0.02,
                'prob': 1.0,
            },
            'coeff_schedule_param': {
                'half_life': 50000,
                'initial_coeff': 0.5,
                'final_coeff': 1,
            },
        }
    },
}