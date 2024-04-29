import os
import numpy as np
from deeputil import Dummy
import plyvel

DUMMY_LOG = Dummy()

# export DISKDICT_DATADIR_PATH=='/home/usr/diskdict_dir_path'
DATA_FILE = os.environ.get('DISKDICT_DATADIR_PATH', '/tmp')


class DiskDict(object):
    # FIXME: using eval - dangerous

    def __init__(self, path, log=DUMMY_LOG):
        '''
        >>> dd = DiskDict(DATA_FILE)
        >>> dd['deepcompute'] = 1
        '''

        self._path = path
        self.log = log
        self._f = plyvel.DB(path, create_if_missing=True, lru_cache_size=1*1024*1024*1024, write_buffer_size=512*1024*1024)

    def _enckey(self, k):
        return k.encode('utf8')

    def _deckey(self, k):
        return k.decode('utf8')

    def get(self, k, default=None):
        '''
        >>> dd = DiskDict(DATA_FILE)
        >>> print(dd.get('deepcompute'))
        1
        '''
        k = self._enckey(k)
        v = self._f.get(k, None)
        if v is None: return default
        return self._decode_value(v)


    def _encode_value(self, v):
        v = np.array(v, dtype='float32')
        head = np.array(v.shape, dtype='int32').tobytes()
        assert len(head) == 8
        content = v.tobytes()
        return head + content


    def _decode_value(self, v):
        shape = np.frombuffer(v[:8], dtype='int32')
        content = np.frombuffer(v[8:], dtype='float32')
        return content.reshape(shape)


    def __contains__(self, k):
        k = self._enckey(k)
        v = self._f.get(k, None)
        return v is not None


    def __getitem__(self, k):
        return self.get(k)


    def __setitem__(self, k, v):
        k = self._enckey(k)
        self._f.put(k, self._encode_value(v))


    def __delitem__(self, k):
        k = self._enckey(k)
        self._f.delete(k)

    def items(self):
        '''
        >>> dd = DiskDict(DATA_FILE)
        >>> print(next(dd.items()))
        ('deepcompute', 1)
        '''

        for k, v in self._f:
            yield self._deckey(k), self._decode_value(v)

    def values(self):
        '''
        >>> dd = DiskDict(DATA_FILE)
        >>> print(next(dd.values()))
        1
        '''

        for _, v in self.items():
            yield v

    def keys(self):
        '''
        >>> dd = DiskDict(DATA_FILE)
        >>> print(next(dd.keys()))
        deepcompute
        '''

        for k, _ in self.items():
            yield k

    def flush(self):
        '''
        >>> dd = DiskDict(DATA_FILE)
        >>> dd.flush()
        '''

        pass

    def close(self):
        '''
        >>> dd = DiskDict(DATA_FILE)
        >>> dd.close()
        '''

        self._f.close()
        self._f = None
