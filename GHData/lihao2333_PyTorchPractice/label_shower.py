class LabelShower(object):

    def __init__(self):
        self._classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def show(self, title, labels):
        print(title, end=" ")
        print(" ".join(self._classes[label] for label in labels))
