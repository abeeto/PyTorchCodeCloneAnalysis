import json
import struct
from functools import lru_cache

import torch
import numpy

CODES_LEN = 4
ENCODING = "utf-8"


@lru_cache(maxsize=128)
def zero_bytes(legnth: int):
    return b'0'*legnth


def jsonpack(data: dict, maxlen: int):
    raw_bytes = json.dumps(data).encode(encoding=ENCODING)
    raw_len = len(raw_bytes)

    codes = struct.pack("<I", raw_len)
    codes_len = len(codes)

    if raw_len + codes_len > maxlen:
        raise ValueError(f"The length of raw data ({raw_len+codes_len}) is larger than limited value ({maxlen}).")

    out = codes + raw_bytes + zero_bytes(maxlen-raw_len-codes_len)

    return out


def jsonunpack(raw_data: bytes):
    if isinstance(raw_data, torch.Tensor):
        raw_data = raw_data.numpy()
    if isinstance(raw_data, numpy.ndarray):
        raw_data = raw_data.tobytes()

    codes = raw_data[:CODES_LEN]
    raw_len, *_ = struct.unpack("<I", codes)
    raw_data: bytes = raw_data[CODES_LEN:CODES_LEN+raw_len]

    data = json.loads(raw_data.decode(encoding=ENCODING))

    return data
