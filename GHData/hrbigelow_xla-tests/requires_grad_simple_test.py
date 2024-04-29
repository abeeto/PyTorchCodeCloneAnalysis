import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl


class RandDataset(torch.utils.data.IterableDataset):
    def __init__(self):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        return torch.rand(10, 10, requires_grad=True)


class RandDataLoader(torch.utils.data.DataLoader):
    @staticmethod
    def ident(x):
        return x

    def __init__(self, dataset):
        super(RandDataLoader, self).__init__(
                dataset=dataset,
                batch_sampler=None,
                collate_fn=self.ident
                )


class TPULoaderIter(object):
    def __init__(self, device):
        dataset = RandDataset()
        data_loader = RandDataLoader(dataset)
        para_loader = pl.ParallelLoader(data_loader, [device])
        self.per_dev_loader = para_loader.per_device_loader(device)

    def __next__(self):
        return self.per_dev_loader.__next__()[0]


def main():
    tpu_iter = TPULoaderIter(xm.xla_device())
    a = next(tpu_iter)
    # FAILS
    assert a.requires_grad
    


if __name__ == '__main__':
    main()

