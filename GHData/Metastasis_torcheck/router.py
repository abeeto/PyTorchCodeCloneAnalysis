from netfilterqueue import NetfilterQueue
from dpkt.ip import IP
from ipaddress import ip_address
from utils import inet_to_str, str_to_inet
import random

LIBNETFILTER_QUEUE_NUM = 1

KNOWN_PEERS = [
    # '10.0.10.4',
    '10.0.10.6',
    '10.0.10.7'
]

TRACKED_CLIENTS = [
    '10.0.10.5',
    '10.0.0.21'
]

TOR_PEERS = [
    '10.0.10.4',
    '10.0.10.6',
    '10.0.10.7'
]


def is_tor(dpkt_ip):
    dst_ip = str(ip_address(dpkt_ip.dst))
    if dst_ip in TOR_PEERS:
        return True
    return False


def update_cksum(dpkt_ip):
    dpkt_ip.sum = 0
    return bytes(dpkt_ip)


def ingress_loop(packet):
    raw_packet = packet.get_payload()
    network = IP(raw_packet)
    src_ip = inet_to_str(network.src)
    dst_ip = inet_to_str(network.dst)
    if src_ip not in TRACKED_CLIENTS:
        return packet.accept()
    if not is_tor(network):
        return packet.accept()
    print('tracked client trying connect to tor')
    network.rf = 1
    if dst_ip in KNOWN_PEERS:
        raw_packet = update_cksum(network)
        packet.set_payload(raw_packet)
        return packet.accept()
    peer = random.choice(KNOWN_PEERS)
    network.dst = str_to_inet(peer)
    raw_packet = update_cksum(network)
    packet.set_payload(raw_packet)
    return packet.accept()


if __name__ == "__main__":
    nfqueue = NetfilterQueue()
    nfqueue.bind(LIBNETFILTER_QUEUE_NUM, ingress_loop)

    try:
        print("[*] waiting for data")
        nfqueue.run()
    except KeyboardInterrupt:
        print("Terminated")

    nfqueue.unbind()
