import os
import logging

class Config:

    embedding_dir = "/data/Images/cifar100/source/embeddings"

    train_image_dir = "/data/Images/cifar100/source/train/images"
    train_image_list = "/data/Images/cifar100/source/train/image.list"

    test_image_dir = "/data/Images/cifar100/source/test/images"
    test_image_list = "/data/Images/cifar100/source/test/image.list"

    # pretrained = False
    # seed = 0
    # num_classes = 100

    # milestones = [60, 120, 160]
    # epochs = 200
    # batch_size = 128
    # accumulation_steps = 1
    # lr = 0.1
    # gamma = 0.2
    # momentum = 0.9
    # weight_decay = 5e-4
    # num_workers = 4
    # print_interval = 30

def setup_logger(logger_name, phase, level=logging.INFO, tofile=False):
    if tofile:
        loger_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), '{}.log'.format(phase))
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh = logging.FileHandler(loger_file, mode='w')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')