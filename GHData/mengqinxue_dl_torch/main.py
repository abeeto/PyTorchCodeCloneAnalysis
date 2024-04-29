from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':
    # create a folder name logs to store historical info
    # in terminal, tensorboard --logdir=logs --port =6007
    writer = SummaryWriter("logs")

    for i in range(100):
        writer.add_scalar("y=x", i, i)
        writer.add_scalar("y=2x", i, 2 * i)
    # writer.add_image()
    # writer.add_scalar()
    writer.close()
