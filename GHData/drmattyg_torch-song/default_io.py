#!/usr/bin/env python

import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import yaml

stream = open('conf/default.yml', 'r')
config = yaml.load(stream)

print('Attempting to default IO to off-state')

try:
    from torch_song.hardware import Igniter
    from torch_song.hardware import Valve
    from torch_song.hardware import MotorDriver
    from torch_song.hardware import PCA9685
except Exception:
    sys.exit(0)

print('Disabling igniters')
for v in config['subsystems']['igniters']:
    try:
        igniter = Igniter(v['gpio'])
        igniter.set_state(0)
    except Exception as e:
        pass

print('Disabling valves')
for v in config['subsystems']['valves']:
    try:
        valve = Valve(v['gpio'])
        valve.set_state(0)
    except Exception as e:
        pass

print('Disabling motors')
for m in config['subsystems']['motors']:
    try:
        motor = MotorDriver(PCA9685(), m['pwm_io'], m['dir_io'], m['dir_io_type'])
        motor.stop()
    except Exception as e:
        pass

