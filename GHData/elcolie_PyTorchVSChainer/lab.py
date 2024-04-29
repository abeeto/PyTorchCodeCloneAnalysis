import chainer as ch
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import datasets
from chainer import training
from chainer.training import extensions


# import matplotlib
# matplotlib.use('Agg')

def network(n_units: int, n_out: int):
    """Network definition."""
    layer = ch.Sequential(L.Linear(n_units), F.relu)
    model = layer.repeat(2)
    model.append(L.Linear(n_out))

    return model

class MLP(ch.Chain):
    """MLP version class based."""
    def __init__(self, input_size: int, n_classes: int):
        super().__init__()
        with self.init_scope():
            self.layer = ch.Sequential(
                L.Linear(input_size), F.relu,
                L.Linear(input_size), F.relu,
                L.Linear(n_classes)
            )
    def forward(self, x):
        h = self.layer(x)
        return h


# Definition of a recurrent net for language modeling
class SimpleGRU(ch.Chain):
    """GRU definition."""
    def __init__(self, input_size: int, n_classes: int):
        super().__init__()
        with self.init_scope():
            self.gru = L.GRU(input_size, n_classes)
            # self.fc1 = ch.Sequential(
            #     L.Linear(500, 500),
            #     F.relu,
            #     L.Linear(500, n_classes)
            # )

        for param in self.params():
            param.array[...] = np.random.uniform(-0.1, 0.1, param.shape)

    def reset_state(self):
        self.gru.reset_state()

    def forward(self, x):
        output = self.gru(x)
        # y = self.fc1(output)
        return output


def main() -> None:
    """Run the code."""
    mushroomsfile = 'mushrooms.csv'
    data_array = np.genfromtxt(
        mushroomsfile, delimiter=',', dtype=str, skip_header=1)
    for col in range(data_array.shape[1]):
        data_array[:, col] = np.unique(data_array[:, col], return_inverse=True)[1]

    X = data_array[:, 1:].astype(np.float32)
    Y = data_array[:, 0].astype(np.int32)[:, None]
    train, test = datasets.split_dataset_random(
        datasets.TupleDataset(X, Y), int(data_array.shape[0] * .7))

    train_iter = ch.iterators.SerialIterator(train, 1)
    test_iter = ch.iterators.SerialIterator(
        test, 1, repeat=False, shuffle=False)

    model = L.Classifier(
        SimpleGRU(22, 1), lossfun=F.sigmoid_cross_entropy, accfun=F.binary_accuracy)

    # Setup an optimizer
    # optimizer = ch.optimizers.SGD().setup(model)
    optimizer = ch.optimizers.Adam().setup(model)

    # Create the updater, using the optimizer
    updater = training.StandardUpdater(train_iter, optimizer, device=-1)

    # Set up a trainer
    trainer = training.Trainer(updater, (1, 'epoch'), out='result')

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=-1))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.DumpGraph('main/loss'))

    trainer.extend(extensions.snapshot(), trigger=(10, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    # Save two plot images to the result dir
    trainer.extend(
        extensions.PlotReport(['main/loss', 'validation/main/loss'],
                              'epoch', file_name='loss.png'))
    trainer.extend(
        extensions.PlotReport(
            ['main/accuracy', 'validation/main/accuracy'],
            'epoch', file_name='accuracy.png'))

    # Print selected entries of the log to stdout
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    #  Run the training
    trainer.run()

    x, t = test[np.random.randint(len(test))]

    predict = model.predictor(x[None]).array
    predict = predict[0][0]

    if predict >= 0:
        print('Predicted Poisonous, Actual ' + ['Edible', 'Poisonous'][t[0]])
    else:
        print('Predicted Edible, Actual ' + ['Edible', 'Poisonous'][t[0]])


if __name__ == "__main__":
    main()
