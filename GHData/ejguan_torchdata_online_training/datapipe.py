import socket
import time
import threading

import torch
import torch.distributed as dist

from concurrent.futures import ThreadPoolExecutor, Future
from torchdata.datapipes.iter import IterDataPipe
from typing import Callable, Optional

_DEFAULT_TIMEOUT = 30 * 60
_DEFAULT_CHECK_INTERVAL = 0.01

__all__ = ["FullSyncIterDataPipe", "OnlineReceiverIterDataPipe"]


class PrefetchExecutor:
    def __init__(
        self,
        datapipe_iterator,
        callback_fn: Optional[Callable[[Future], None]] = None,
        timeout: int = _DEFAULT_TIMEOUT,
    ) -> None:
        self.datapipe_iterator = datapipe_iterator
        self.callback_fn = callback_fn
        self.timeout = timeout
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._get_data_future: Future = self._executor.submit(self.fetch_next)
        if self.callback_fn is not None:
            self._get_data_future.add_done_callback(self.callback_fn)

    def fetch_next(self):
        return next(self.datapipe_iterator)

    def return_next(self):
        data = self._get_data_future.result(timeout=self.timeout)
        self._get_data_future = self._executor.submit(self.fetch_next)
        if self.callback_fn is not None:
            self._get_data_future.add_done_callback(self.callback_fn)
        return data

    def shutdown(self):
        self._executor.shutdown(wait=True)


class FullSyncIterDataPipe(IterDataPipe):
    def __init__(self, datapipe, timeout=_DEFAULT_TIMEOUT):
        self.datapipe = datapipe
        self.timeout = timeout

        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError(
                "Torch Distributed is required to be initialized"
            )
        self._process_group = dist.new_group(backend="gloo")
        self._world_size = dist.get_world_size()
        self._total_cnt = torch.tensor([0], dtype=torch.int32)

        self._lock = threading.RLock()
        self._cv = threading.Condition(lock=self._lock)
        self._executor = None
        self._error = None
        self._finished_callback = False

    def _callback_fn(self, f: Future) -> None:
        with self._cv:
            if f.exception() and not isinstance(f.exception(), StopIteration):
                self._error = f.exception()
            else:
                if isinstance(f.exception(), StopIteration):
                    self._total_cnt = torch.tensor([0], dtype=torch.int32)
                else:
                    self._total_cnt = torch.tensor([1], dtype=torch.int32)
            dist.all_reduce(
                tensor=self._total_cnt,
                op=dist.ReduceOp.SUM,
                group=self._process_group,
            )
            self._finished_callback = True
            self._cv.notify()

    def __iter__(self):
        assert self._executor is None
        self._executor = PrefetchExecutor(
            iter(self.datapipe),
            self._callback_fn,
            self.timeout
        )
        data = -1
        while True:
            with self._cv:
                is_success = self._cv.wait_for(
                    lambda: self._finished_callback is True,
                    self.timeout,
                )
                if not is_success:
                    raise RuntimeError("Timeout")
                if self._error is not None:
                    raise self._error
                if bool(self._total_cnt < self._world_size):
                    break
            data = self._executor.return_next()
            self._finished_callback = False
            yield data

    def reset(self):
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None
        self._total_cnt = torch.tensor([0], dtype=torch.int32)
        self._error = None
        self._finished_callback = False


class OnlineReceiverIterDataPipe(IterDataPipe):
    def __init__(self, timeout=_DEFAULT_TIMEOUT, buffer_size=1024):
        self.timeout = timeout
        self.buffer_size = buffer_size
        self._host = None
        self._port = None
        self._local_rank = 0
        self._local_rank_str = "0"
        self._conn = None

    def set_connection_config(self, host, port, local_rank=0):
        self._host = host
        self._port = port
        self._local_rank = local_rank
        self._local_rank_str = str(local_rank)

    def __iter__(self):
        assert self._conn is None
        self._conn = socket.create_connection(
            (self._host, self._port),
            self.timeout,
        )
        self._conn.send(
            f"Try to connect from local rank {self._local_rank_str}"
            .encode("utf-8")
        )
        buffer = b""
        # TODO: Add timeout here
        while len(buffer) < 9:
            resp = self._conn.recv(self.buffer_size).decode("utf-8")
            if resp:
                buffer += resp
            else:
                time.sleep(_DEFAULT_CHECK_INTERVAL)

        resp = buffer[:9]
        buffer = buffer[9:]
        assert resp == "Connected"

        data_size = 0
        while True:
            while True:
                resp = self._conn.recv(self.buffer_size)
                if resp:
                    buffer += resp
                else:
                    time.sleep(_DEFAULT_CHECK_INTERVAL)
                # Use 4 bytes as the meta data to indicate size of each data
                if len(buffer) > 4:
                    data_size = int.from_bytes(buffer[:4], byteorder='big')
                    buffer = buffer[4:]
                if data_size > 0 and len(buffer) >= data_size:
                    data = buffer[:data_size]
                    buffer = buffer[data_size:]
                    data_size = 0
                    break
            yield data
        self._conn.close()

    def reset(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None
