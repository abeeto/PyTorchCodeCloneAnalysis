from tensorboardX import SummaryWriter

if __name__ == '__main__':
    # x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    writer = SummaryWriter('./log_test/')
    for i in range(1, 10):
        writer.add_scalar('test', i, i)
    writer.close()
