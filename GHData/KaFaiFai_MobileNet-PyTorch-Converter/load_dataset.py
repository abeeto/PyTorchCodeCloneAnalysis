"""
Load dataset to see if any error occurs
"""
import argparse
import timeit

from torch.utils.data import DataLoader, SequentialSampler

from config import DATASETS

parser = argparse.ArgumentParser()
# required settings
parser.add_argument("--data", type=str, help="dataset", choices=DATASETS.keys(), required=True)
parser.add_argument("--batch-size", type=int, help="loading batch size", required=True)

# misc settings
parser.add_argument("--is-test", action='store_true', help="Load the test dataset if specified")
parser.add_argument("--print-step", type=int, help="How often to print progress (in batch)?")
parser.add_argument("--start", type=int, help="Which batch to start from?")
parser.add_argument("--end", type=int, help="Which batch to end in?")


class PartialSequentialSampler(SequentialSampler):
    """
    to only load batches in the middle
    """

    def __init__(self, data_source, start=None, end=None, batch_size=1):
        super().__init__(data_source)
        num_batch = (len(data_source) - 1) // batch_size + 1
        self.start = 0 if start is None else start
        self.end = num_batch if end is None else end
        self.batch_size = batch_size

        if self.start > self.end:
            raise Exception(f"Start index {self.start} > end index {self.end}")
        assert self.start >= 0
        assert self.end <= num_batch, f"total number of batch = {num_batch}, but got {self.end}"
        assert self.batch_size >= 1

    def __iter__(self):
        return iter(range(self.start * self.batch_size, min(self.end * self.batch_size, len(self.data_source))))


def load_dataset(configs):
    start = timeit.default_timer()

    Dataset = DATASETS[configs.data]["class"]
    root_dir = DATASETS[configs.data][("test" if configs.is_test else "train") + "_root"]
    dataset = Dataset(root=root_dir, is_train=not configs.is_test)

    print_step = configs.print_step
    batch_size = configs.batch_size
    start_idx = configs.start
    end_idx = configs.end

    sampler = PartialSequentialSampler(dataset, start_idx, end_idx, batch_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=sampler)

    mid = timeit.default_timer()
    print(f"Time for reading dataset: {mid - start:.2f}s")
    last_batch = -1
    try:
        for batch_idx, _ in enumerate(dataloader):
            if print_step is not None and batch_idx % print_step == 0:
                print(f"Loading [Batch {batch_idx + sampler.start:4d}/{len(dataloader)}] ...")
            last_batch = batch_idx
    except Exception as e:
        print(f"Loading batch {last_batch + 1} with batch_size={batch_size} when error occurs")
        print(e)

    end = timeit.default_timer()
    num_batch = sampler.end - sampler.start - 1
    print(f"Time for iterating {num_batch} batches with batch_size={batch_size}: {end - mid:.2f}s | "
          f"{(end - mid) / num_batch:.2f}s/batch")


if __name__ == '__main__':
    args = parser.parse_args()
    load_dataset(args)
