import requests

def _parse_nodes(exit_addresses):
    is_exit_address = lambda s: s.startswith("ExitAddress")
    return {line.split()[1]
            for line in exit_addresses.split('\n')
            if is_exit_address(line)}

_EXIT_NODES = _parse_nodes(requests.get("https://check.torproject.org/exit-addresses").text)

def is_exit_node(ip):
    return ip in EXIT_NODES
