import os, configparser, random, logging
import torch
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


# 控制random
def seed_setting(seed_number):
    random.seed(seed_number)
    os.environ['PYTHONHASHSEED'] = str(seed_number)
    np.random.seed(seed_number)
    torch.manual_seed(seed_number)
    torch.cuda.manual_seed(seed_number)
    torch.cuda.manual_seed_all(seed_number)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


# visualization
def get_summary_writer(log_filepath):
    """
    在终端输入： tensorboard --logdir=log_filepath --port XXX
    e.g. tensorboard --logdir=run --port 4444
    """
    time_stamp = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    log_dir = os.path.join(log_filepath, time_stamp)
    log_writer = SummaryWriter(log_dir=log_dir)
    return log_writer, log_dir


# 保存模型
def save_model(model, save_filepath):
    torch.save(model, save_filepath)


# 加载模型
def load_model(load_filepath):
    model = torch.load(load_filepath)
    return model


# 讲模型的训练参数加以保存
def record_configuration(save_dir, configuration_dict: dict):
    file_name = os.path.join(save_dir, 'configuration.ini')
    write_config = configparser.ConfigParser()
    for config_key, config_value in configuration_dict.items():
        write_config.add_section(config_key)
        for sub_config_key, sub_config_value in config_value.items():
            write_config.set(config_key, sub_config_key, str(sub_config_value))
    # save configuration
    cfg_file = open(file_name, 'w')
    write_config.write(cfg_file)
    cfg_file.close()


def init_logger(log_file):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)

    handler = logging.FileHandler(log_file, mode='w')
    handler.setLevel(logging.DEBUG)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    logger.addHandler(handler)
    logger.addHandler(console)
    return logger