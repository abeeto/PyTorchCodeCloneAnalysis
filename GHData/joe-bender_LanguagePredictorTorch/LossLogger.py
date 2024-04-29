class LossLogger:
    # read tuples of train_loss and test_loss
    def read_losses(self, filename):
        with open(filename) as f:
            pairs = f.read()
            pairs = pairs.split('\n')[:-1]
            pairs = [pair.split(',') for pair in pairs]
            pairs = [(float(pair[0]), float(pair[1])) for pair in pairs]
            train_losses, test_losses = zip(*pairs)
            return train_losses, test_losses

    # write tuple of train_loss and test_loss
    def write_losses(self, train_loss, test_loss, filename):
        with open(filename, 'a') as f:
            f.write('{},{}\n'.format(train_loss, test_loss))

    def clear(self, filename):
        with open(filename, 'w') as f:
            pass
