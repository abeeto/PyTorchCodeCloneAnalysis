# number of test images
TESTS = 1
# images are sorted in folder so a specific start idx can be used
IMG_IDX = 0

# only sample style do not change
SAMPLE_STYLE = 'conditional'

# feature size
WIN_SIZE = 20 

OVERLAPPING = 'stride' #'stride' 'None' 'full'
STRIDE      = 5 # only for stride
NUM_SAMPLES = 10
PADDING_SIZE = 2
# Imagenet 
#   'resnet50', 'mnv2', 'vgg16', 'alexnet'
# costum rsicd
#   'c_jakob' 'c_conrad
MODEL = 'mnv2'


SOFTMAX = True
OOD     = False # set to false for all imagenet models!

SHOW = False  # show  after processing


# gpu batch size
BATCH_SIZE = 64

#-------------------
# LAPLACEN CORRECTION PDA IMAGENET (DO NOT CHANGE!)
TRAINSIZE = 100000
CLASSES   = 1000

# RSICD (does not change anything!)
# TRAINSIZE = 10921


# DATASETS:
# 'so2satlcz42' (10), 'xview2'(7), 'rsicd'(23), 'sen12ms'(9), 'mnist_fashion'(7), 'cifar10'(6) ', 'ilsvrc_2012'(1000)
DSET = 'ilsvrc_2012_test'
CLASSES   = 1000


# REGRESSION MODEL SELECTION
#'jakob', 'rabby', 'conrad', 'samuel'
NAME = '' # _conrad_' # '' 
# 'regression_' or '' for no regression 
REGRESSION = '' # 'regression_' 
# comment in to not use regression models!!
# REGRESSION = ''
if REGRESSION != '':
    CLASSES = 1

# image net
# CATEGORIE_PATH = './model/ilsvrc_2012_labels.txt'
CATEGORIE_PATH = './model/' + DSET + '_labels.txt'

if OOD or REGRESSION != '':
    # no real lables, dummy for ood score
    CATEGORIE_PATH = './model/ood_lables.txt'


#--- SELECT DATA SOURCE ---#
DATASET_PATH = './data/' + DSET + '/' 


#*** TEST INET    
IMAGE_PATH = DATASET_PATH


RESULT_PATH = './res_' + REGRESSION +  DSET + '{}'.format(NAME) +  str(int(OOD)) +  '/'

# number of images for statistic paramter of conditional sampler
# IMAGENET : 512, IMAGENET_TEST : 18, HTCV Testsets: 100
PARAM_DATASET_SIZE = 16

