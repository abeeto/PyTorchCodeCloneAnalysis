import collections
from typing import Dict
class FrozenDict(collections.Mapping):

    def __init__(self, d: Dict):
        self._d = dict()
        # deep copy because like it seems like the right idea
        for k,v in d.items():
            self._d[k] = v
        self._hash = None

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)

    def __getitem__(self, key):
        return self._d[key]

    def __hash__(self):

        if self._hash is None:
            hash_ = 0
            for pair in self.items():
                hash_ ^= hash(pair)
            self._hash = hash_
        return self._hash