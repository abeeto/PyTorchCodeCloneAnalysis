EPOCHS = 75
SAVE_EVERY = 5 # after how many epochs to save a checkpoint
LOG_EVERY = 5 # log training and validation metrics every `LOG_EVERY` epochs
BATCH_SIZE = 16
DEVICE = 'cuda'
LR = 0.0001
ROOT_PATH = '../input/camvid'

# the classes that we want to train
CLASSES_TO_TRAIN = [
        'animal', 'archway', 'bicyclist', 'bridge', 'building', 'car', 
        'cartluggagepram', 'child', 'columnpole', 'fence', 'lanemarkingdrve', 
        'lanemarkingnondrve', 'misctext', 'motorcyclescooter', 'othermoving',
        'parkingblock', 'pedestrian', 'road', 'road shoulder', 'sidewalk',
        'signsymbol', 'sky', 'suvpickuptruck', 'trafficcone', 'trafficlight', 
        'train', 'tree', 'truckbase', 'tunnel', 'vegetationmisc', 'void',
        'wall'
        ]

# DEBUG for visualizations
DEBUG = False