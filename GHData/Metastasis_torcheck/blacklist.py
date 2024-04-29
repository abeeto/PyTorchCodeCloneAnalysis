from config import BLACKLIST_PATH
from utils import is_valid_ipv4_address


class Blacklist:
    def __init__(self):
        self.data = {}
        self.ips = {}

    def load(self, pathname=None):
        path = pathname

        if not path:
            path = BLACKLIST_PATH

        with open(path, 'r') as f:
            for line in f.read().splitlines():
                if not line:
                    continue

                url, ip_list = self._parse(line)

                self.data[url] = ip_list
                for ip in ip_list:
                    self.ips[ip] = True

    def save(self):
        if not len(self.data.keys()):
            raise ValueError("Config is empty. Nothing to save")
        lines = ""
        for url, ip_list in self.data.items():
            lines = lines + "{};{}\n".format(url, ','.join(ip_list))
        with open(BLACKLIST_PATH, 'w+') as f:
            f.write(lines)

    def _parse(self, line):
        url, all_ip = line.split(';')
        ip_list = all_ip.split(',')

        return url, ip_list

    def _unpack(self, item):
        pass

    def __len__(self):
        return len(self.data)

    def __contains__(self, key):
        if is_valid_ipv4_address(key):
            return key in self.ips
        return key in self.data

    def has(self, key):
        return self.__contains__(key)

# if __name__ == '__main__':
#     blacklist = Blacklist()
#     blacklist.load()
#     print('parimatch-go2.com' in blacklist.data)
#     print('104.28.0.94' in blacklist.data)
#     print('parimatch-go2.com' in blacklist)
#     print('104.28.0.94' in blacklist)
#     print(blacklist.has('parimatch-go2.com'))
#     print(blacklist.has('104.28.0.94'))
#     print('parimatc.com' in blacklist.data)
#     print('parimatc.com' in blacklist)
#     print(blacklist.has('104.28.3.94'))
