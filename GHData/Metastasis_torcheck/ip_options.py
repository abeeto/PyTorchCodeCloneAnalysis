from dpkt import Packet, in_cksum
from dpkt.ip import IP


class IPOption(Packet):
    """Internet Protocol Options.

    Attributes:
        __hdr__: Option fields of IP.
    """

    __hdr__ = (
        ('type', 'B', 0),
        ('length', 'B', 0)  # make it optionals, cause not all options have this field
    )

    def __init__(self, *args, **kwargs):
        super(IPOption, self).__init__(*args, **kwargs)

    @property
    def copied(self):
        return self.type >> 7

    @copied.setter
    def copied(self, copied):
        self.type = ((copied & 128) << 7) | self.type

    @property
    def category(self):
        return (self.type & 96) >> 5

    @category.setter
    def category(self, category):
        self.type = ((category & 96) << 5) | self.type

    @property
    def number(self):
        return self.type & 31

    @number.setter
    def number(self, number):
        self.type = (number & 31) | self.type

    def __len__(self):
        if self.length:
            return self.length
        return self.__hdr_len__ + len(self.data)

    def __bytes__(self):
        return self.pack_hdr() + bytes(self.data)

    def unpack(self, buf):
        Packet.unpack(self, buf)
        self.data = buf[self.__hdr_len__:self.length + 1]


class IPOptions(IPOption):
    def unpack(self, buf):
        # TODO: create algorithm for options
        IPOption.unpack(self, buf)
        self.data = buf[self.__hdr_len__:self.length + 1]


def append_options(ip, new_option):
    has_options = len(ip.opts) > 0
    if has_options:
        return ip

    DWORD = 4  # bytes
    EOL_LEN = 1  # DWORD
    option_eol = b'\x00'  # End of Options List
    opts_len = int(new_option.length / DWORD) + EOL_LEN  # DWORDS
    header_len = ip.hl + opts_len
    if header_len > 15:
        return ip

    opts_len = opts_len * DWORD  # bytes

    ip.hl = header_len
    ip.len = ip.len + opts_len
    ip.opts = bytes(new_option) + option_eol

    padding_len = opts_len - len(ip.opts)

    ip.opts = ip.opts + (b'\x00' * padding_len)
    ip.sum = 0

    return ip


def test_opt():
    s = b'\x4f\x00\x00\x3c\xae\x08\x00\x00\x40\x06\x18\x10\xc0\xa8\x0a\x26\xc0\xa8\x0a\x01\x87\x27\x08\x01\x02\x03\x04\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06\x05\x04\x03\x02\x01'
    ip = IP(s)
    opts = IPOption(ip.opts)
    assert (bytes(opts) == ip.opts)
    opts.length = 12
    assert (opts.length == 12)
    opts.copy = 1
    assert (opts.copy == 1)
    raw_opts = b'\x44\x0c\x05\x01\x00\x00\x00\x00\x00\x00\x00\x00'
    new_opts = IPOption(
        type=0x44,
        length=0x0c,
        data=b'\x05\x01\x00\x00\x00\x00\x00\x00\x00\x00'
    )
    assert (raw_opts == bytes(new_opts))


if __name__ == '__main__':
    test_opt()
    print('Tests Successful...')
