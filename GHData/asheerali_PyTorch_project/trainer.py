import torch as t
from sklearn.metrics import f1_score


class EarlyStop:
    def __init__(self, early_stopping_patience=100000):
        self.early_stopping_patience = early_stopping_patience
        self.save_loss = 1000
        self.sum = -1

    def step(self, validation_loss):
        if validation_loss < self.save_loss:
            self.save_loss = validation_loss
            self.sum += 0
        else:
            self.sum += 1

        if self.sum > self.early_stopping_patience:
            return True
        else:
            return False


class Trainer:
    def __init__(
        self,
        model,
        crit,
        optim=None,
        train_dl=None,
        val_test_dl=None,
        cuda=True,
        early_stopping_patience=20
    ):
        self.model = model
        self.crit = crit
        self.optim = optim
        self.train_dl = train_dl
        self.val_test_dl = val_test_dl
        self.cuda = cuda
        self.early_stopping_patience = early_stopping_patience
        self.early_stop = EarlyStop()

        if cuda:
            self.model = model.cuda()
            self.crit = crit.cuda()

        self.counter = 0

    def save_checkpoint(self, epoch):
        t.save({'state_dict': self.model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))

    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self.cuda else None)
        self.model.load_state_dict(ckp['state_dict'])

    def save_onnx(self, fn):
        m = self.model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self.model(x)
        t.onnx.export(
            m,
            x,
            fn,
            export_params=True,
            opset_version=10,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {
                    0: 'batch_size'
                },
                'output': {
                    0: 'batch_size'
                }
            }
        )

    def train_step(self, x, y):
        self.optim.zero_grad()
        output = self.model(x)
        loss = self.crit(output, y)
        loss.backward()
        self.optim.step()
        return loss

    def val_test_step(self, x, y):
        output = self.model(x)
        loss = self.crit(output, y)
        output[output >= 0.5] = 1
        output[output < 0.5] = 0
        return loss, output

    def train_epoch(self):
        self.model.train()
        for epoch in range(2):
            running_loss = 0.0
            for i, (x, y) in enumerate(self.train_dl):
                if self.cuda:
                    x = x.cuda()
                    y = y.cuda()

                running_loss += self.train_step(x, y).item()
        return running_loss / (self.train_dl.__len__())

    def val_test(self):
        self.model.eval()
        running_loss = 0.0
        inactive_len = 0
        inactive_sum = 0
        crack_len = 0
        crack_sum = 0
        pred_list = []
        y_true = []
        y_pred = []
        with t.no_grad():
            for i, (x, y) in enumerate(self.val_test_dl):
                if self.cuda:
                    x = x.cuda()
                    y = y.cuda()
                loss, pred = self.val_test_step(x, y)
                pred = pred.cpu()
                y = y.cpu()
                for m in y:
                    y_true.append(transform1d(m))

                for m in pred:
                    y_pred.append(transform1d(m))
                running_loss += loss.item()
                average_loss = running_loss / self.val_test_dl.__len__()

            for pred_batch in pred_list:
                for pred in pred_batch:
                    y_pred.append(transform1d(pred))

            for i in range(len(y_pred)):
                if y_true[i] == 1 or y_true[i] == 3:
                    inactive_len += 1
                    if y_pred[i] == y_true[i]:
                        inactive_sum += 1
                if y_true[i] == 2 or y_true[i] == 3:
                    crack_len += 1
                    if y_pred[i] == y_true[i]:
                        crack_sum += 1

            mean_F1_score = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
            crack_accuracy = crack_sum / inactive_len
            inactive_accuracy = inactive_sum / inactive_len
            print("crack_accuracy: %.3f%%" % (crack_accuracy * 100))
            print("inactive_accuracy: %.3f%%" % (inactive_accuracy * 100))
            print("mean_F1_score %.2f" % (mean_F1_score))
        return average_loss

    def fit(self, epochs):
        assert self.early_stopping_patience > 0 or epochs > 0

        train_loss_list = []
        validation_loss_list = []

        while True:
            if self.counter >= epochs:
                return train_loss_list, validation_loss_list

            else:

                train_loss = self.train_epoch()
                validation_loss = self.val_test()

                train_loss_list.append(train_loss)
                validation_loss_list.append(validation_loss)

                self.save_checkpoint(self.counter)

                if self.early_stop.step(validation_loss):
                    return train_loss_list, validation_loss_list

                else:
                    self.counter += 1
                    print("counter: ", self.counter)
                    print("elp: ", self.early_stopping_patience)


def transform1d(x):
    if x[0] == 0 and x[1] == 0:
        return 0
    elif x[0] == 0 and x[1] == 1:
        return 1
    elif x[1] == 1 and x[1] == 0:
        return 2
    else:
        return 3
