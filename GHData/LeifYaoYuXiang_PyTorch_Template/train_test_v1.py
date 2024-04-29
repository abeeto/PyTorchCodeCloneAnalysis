import torch


def train_and_eval(model, optimizer, train_dataloader, eval_dataloader,
                   device, train_config, n_epoch, logger, summary_writer):
    train_acc_list = []
    val_acc_list = []
    for each_epoch in range(n_epoch):
        loss, train_acc, val_acc = train_and_eval_in_one_epoch(model, optimizer, train_dataloader, eval_dataloader, train_config, logger)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        logger.info(str(each_epoch) + ' ' + str(loss) + ' ' + str(train_acc) + ' ' + str(val_acc))
        summary_writer.add_scalar('Loss', loss, each_epoch)
        summary_writer.add_scalar('TrainAcc', train_acc, each_epoch)
        summary_writer.add_scalar('ValAcc', val_acc, each_epoch)



def train_and_eval_in_one_epoch(model, optimizer, train_dataloader, eval_dataloader, train_config, logger):
    # 开始训练
    model.train()

    # 开始测试
    with torch.no_grad():
        pass