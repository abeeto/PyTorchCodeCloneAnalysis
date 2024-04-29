# Space Engineers Watchdog
# Copyright (C) 2022  Raelon "th3r00t" Masters
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from pathlib import PureWindowsPath, Path
import configparser
import asyncio
import os
import sys
import signal
import subprocess
import time
from unicodedata import name

config = {}


def mkconfig():
    try:
        _fparser = configparser.ConfigParser()
        _fparser["instance"] = {
            "path": "/path/to/server/Instance/",
            "exe": "Torch.Server.exe",
            "timeout": 60,
            "nexus": False
        }
        with open(r'./sewatchdog.ini', 'w') as _cfile:
            _fparser.write(_cfile)
            _cfile.flush()
            _cfile.close()
        print("Please adjust sewatchdog.ini accordingly")
        sys.exit()
    except Exception as e:
        print(e)
        return False


def getconfig():
    if not Path("./sewatchdog.ini").is_file():
        if not mkconfig():
            print("Error during config file generation")
            return False
    try:
        _fparser = configparser.ConfigParser()
        _fparser.read('./sewatchdog.ini')
        return [_fparser['instance']['path'],
                _fparser['instance']['exe'],
                _fparser['instance']['timeout'],
                _fparser['instance']['nexus']]
    except Exception as e:
        print(e)
        return False


class Server:
    global config

    def __init__(self):
        self.name = config[1]
        self.instance_path = config[0]
        self.server_path = self.instance_path+'../'
        self.is_nexus = config[3]
        self.pid = self.getpid(self.instance_path)
        self.exe = config[1]
        self.last_stamp = None
        self.getcanary()

    def watchdog(self):
        _watcher = True
        while _watcher:
            last_stamp = self.last_stamp
            time.sleep(20)
            self.getcanary()
            if last_stamp is None:
                if self.is_nexus:
                    print("Nexus Configuration Detected, Extending Timeout")
                    time.sleep(420)
                print('Waiting for Game Ready')

                time.sleep(60)
            elif self.last_stamp - last_stamp == 0:
                if time.time() - self.last_stamp >= config[2]:
                    print(self.last_stamp - last_stamp)
                    print(r'Killing Server {}'.format(time.time()))
                    self.die()
                    time.sleep(5)
                    self.spawn()
                else:
                    print(r'{} since last heartbeat'.format(
                        time.time() - self.last_stamp))
            elif self.last_stamp - last_stamp != 0:
                print(r'{} pulse rcvd {}'.format(
                    self.last_stamp - last_stamp, time.time()))

    def die(self):
        print('Killing Server')
        os.system("taskkill /f /im Testing.Server.exe")
        self.pid = None
        self.last_stamp = None

    def spawn(self):
        print("Launching Server")
        _server_path = Path(self.server_path, self.exe).__str__()
        subprocess.Popen(_server_path, close_fds=True,
                         creationflags=subprocess.DETACHED_PROCESS)

    def __str__(self):
        return self.name

    def getcanary(self):
        try:
            _fp = self.instance_path.replace('"', '')+'canary'
            self.last_stamp = os.stat(_fp).st_mtime
            with open(_fp, 'r') as _canary:
                pulse = _canary.readline()
                _canary.close()
            return pulse
        except Exception as e:
            print(e)
            return False

    def getpid(self, server):
        try:
            _fp = server.replace('"', '')+'pid'
            with open(_fp, 'r') as _pid:
                return _pid.readline()
        except FileNotFoundError as e:
            print(e)
            return False


async def main():
    global config
    print('sewatchdog: initializing systems')
    config = getconfig()
    print('sewatchdog: got configuration')
    server = Server()
    print('sewatchdog: initializing watchdog')
    await server.watchdog()
    await asyncio.sleep(.2)

if __name__ == "__main__":
    asyncio.run(main())
    print("Goodbye")
