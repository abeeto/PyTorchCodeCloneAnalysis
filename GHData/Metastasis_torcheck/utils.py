import socket
from dpkt.ip import IP
from random import randrange
import struct


def modify_pkt_rnd(net_packet):
    # net.data is layer 4 packet
    # so net.data.data is layer 5 or just a payload of layer 4
    net = IP(net_packet.get_payload())
    tran_len = len(net.data)

    payload_len = net.len - tran_len
    if not payload_len:
        return net.pack()

    new_data = bytearray(net.data.pack())
    for idx in range(10):
        rnd_byte = randrange(0, payload_len)
        # God, please, i hope there's no off-by-one error
        new_data[(tran_len - payload_len) + rnd_byte] = rnd_byte
        # new_data[rnd_byte] = bytes([rnd_byte])

    net.data = new_data

    return net.pack()


def is_tor_port(port):
    return port == 443 or (port >= 9000 and port <= 9100) or (port >= 8000 and port <= 8100)


def inet_to_str(inet):
    """Convert inet object to a string

        Args:
            inet (inet struct): inet network address
        Returns:
            str: Printable/readable IP address
    """
    try:
        return socket.inet_ntop(socket.AF_INET, inet)
    except ValueError:
        return socket.inet_ntop(socket.AF_INET6, inet)


def str_to_inet(ip_addr):
    """Convert string to int ip

        Args:
            ip_addr: IP address as a string
        Returns:
            int: IP address in integer format
    """
    return socket.inet_aton(ip_addr)


def is_valid_ipv4_address(address):
    """Checks whatever address is valid IPv4 string

        Args:
            address: IP address as a string
        Returns:
            bool: True if address is valid IPv4
    """
    try:
        socket.inet_pton(socket.AF_INET, address)
    except AttributeError:  # no inet_pton here, sorry
        try:
            socket.inet_aton(address)
        except socket.error:
            return False
        return address.count('.') == 3
    except socket.error:  # not a valid address
        return False

    return True


def dump(fp, stream):
    if not fp:
        return None

    per_line = 10
    cnt = 0
    line = ''

    for byte in bytearray(stream):
        char = chr(byte)

        if char == ' ':
            char = '.'

        cnt = cnt + 1
        line = line + char

        if not (cnt % per_line):
            line = line + '\n'
            fp.write(line)
            cnt = 0
            line = ''

    if line:
        line = line + '\n'
        fp.write(line)


def save_connections(connections):
    for flow, stream in connections.items():
        fname = '{}:{}_{}:{}'.format(flow[0], flow[1], flow[2], flow[3])

        with open('./tmp/' + fname, 'w') as fp:
            dump(fp, stream)
