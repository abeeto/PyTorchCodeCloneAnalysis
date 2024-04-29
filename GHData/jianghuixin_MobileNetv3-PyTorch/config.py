class DefaultConfig:
    # visdom 参数
    env = "CatDog"  # visdom 环境, None 表示不使用 visdom 可视化

    # 模型参数
    model = "MobileNetV3"  # 使用的模型, 命名与 models/__init__.py 一致
    load_model_path = None  # 加载预训练模型的路径, None 表示不加载

    # 数据集参数
    data_root = "/home/airice/sdb/datasets/CatDog"  # 该路径包含 "test1" 和 "train" 子文件夹
    batch_size = 128
    num_workers = 16  # 并行加载的进程数

    # 训练参数
    use_gpu = True
    max_epoch = 30
    lr = 1e-3
    lr_decay = 0.95  # 当 val_loss 增大时, lr = lr * lr_decay
    weight_decay = 1e-4  # L2 正则化
