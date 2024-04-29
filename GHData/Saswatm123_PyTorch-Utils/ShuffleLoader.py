from torch.utils.data import Dataset as DSMixin, DataLoader

class ShuffleLoader(DSMixin):
    '''
        Standard torch.utils.data.DataLoader, with reassigned batches every
        run-through to prevent overfitting to one batch distribution.
    '''
    def __init__(self, dataset, batch_size, shuffle = True):
        super().__init__()
        self.loader = torch.utils.data.DataLoader(
            dataset = dataset,
            batch_size = batch_size,
            shuffle = shuffle
        )
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def __iter__(self):
        def next_generator(loader_instance):
            for X in loader_instance.loader:
                yield X
            loader_instance.loader = torch.utils.data.DataLoader(
                dataset = loader_instance.dataset,
                batch_size = loader_instance.batch_size,
                shuffle = loader_instance.shuffle
            )
        return next_generator(self)
