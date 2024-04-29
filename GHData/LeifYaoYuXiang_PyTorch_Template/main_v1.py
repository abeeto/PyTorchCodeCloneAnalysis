import os
import torch
from argument_parse_v1 import parse_args
from dataloader_v1 import generate_dataloader
from model_v1 import Model
from train_test_v1 import train_and_eval
from utils import save_model, seed_setting, get_summary_writer, record_configuration, init_logger


def main(args):
    random_seed = args.random_seed
    seed_setting(random_seed)

    dataset_name = args.dataset_name
    # 加载并且生成可以用来训练的数据
    train_dataloader, eval_dataloader = generate_dataloader(args)
    dataset_config = {
        'root_dir': dataset_name
    }

    # 生成相关的模型
    model_config = {
        'num_layers': args.num_layers,
    }
    model = Model(model_config)

    n_epoch = args.n_epochs
    lr = args.lr
    log_filepath = os.path.join(args.log_filepath, dataset_name)

    device = args.device
    if not torch.cuda.is_available() and device == 'gpu':
        device = 'cpu'

    train_config = {
        'n_epoch': args.n_epochs,
        'lr': args.lr,
        'log_filepath': args.log_filepath,
        'random_seed': args.random_seed,
        'device': args.device,
    }
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    summary_writer, log_dir = get_summary_writer(log_filepath)
    record_configuration(save_dir=log_dir, configuration_dict={
        'MODEL': model_config,
        'DATASET': dataset_config,
        'TRAIN': train_config,
    })
    logger = init_logger(os.path.join(log_dir, 'log.log'))

    # 开始训练和测试
    try:
        model = train_and_eval(model, optimizer,
                               train_dataloader, eval_dataloader,
                               device, train_config, n_epoch, logger, summary_writer)
        return model, log_dir
    except KeyboardInterrupt:
        return model, log_dir


if __name__ == '__main__':
    args = parse_args()
    integrated_model, log_dir = main(args)
    save_model(integrated_model, os.path.join(log_dir, 'model.pkl'))