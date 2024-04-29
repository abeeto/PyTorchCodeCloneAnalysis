'''
Created on Mar 1, 2013

@author: Pat Nystrom
'''
import serial
import modbus as M
from crc16 import block_crc16 as modbus_crc16
import binascii
from struct import pack, unpack, error as struct_error
import time

DEFAULT_PROFILE = [
                   (70, 12),
                   (90, 20),
                   (110, 8),
                   (120, 5),
                   (135, 5),
                   (140, 5),
                   (150, 8),
                   (155, 10),
                   (158, 10),
                   (159, 10),
                   (162, 10),
                   (163, 10),
                   (164, 10),
                   (165, 10),
                   (166, 10),
                   (167, 10),
                   (168, 10),
                   (169, 10),
                   (170, 10),
                   (171, 3),
                   (172, 1),
                   (173, 1),
                   (174, 3),
                   (175, 5),
                   (190, 20),
                   (200, 5),
                   (210, 10),
                   (220, 15),
                   (230, 15),
                   (235, 20),
                   (240, 25),
                   (242, 30),
                   (243, 30),
                   (245, 10),
                   (250, 8),
                   (140, 10),
                   (90, 30),
                   (40, 30),
                   (10, 30),
                   (1, 150),
                  ]

class TorchOven(object):
    def __init__(self, port=0):
        self.sp = serial.Serial(port, baudrate=9600)
        self.sp.setTimeout(0.5)
        self.fmt = M.Formatter(addr=2)

    def cksum_msg(self, cmd):
        return cmd+pack('>H', modbus_crc16(cmd))

    def write_reg(self, reg, value):
        time.sleep(0.06)
        msg = self.cksum_msg(self.fmt.write_single_reg(reg, value))
        print 'writing %.04x v=%d with cmd %s' % (reg, value, binascii.b2a_hex(msg))
        self.sp.write(msg)
        resp = self.sp.read(6) # If this were standard MODBUS, it would be an 8-byte read
        return resp

    def read_regs(self, addr, count):
        ''' read count modbus regs starting at addr.
          return raw serial reply, including crc16 as last two bytes'''
        time.sleep(0.06)
        msg = self.cksum_msg(self.fmt.read_holding_regs(addr, count))
        print 'reading reg %.04x with cmd %s' % (addr, binascii.b2a_hex(msg))
        self.sp.flushInput()
        self.sp.write(msg)
        resp = self.sp.read(count*2 + 5)  # response has 2 byte header + 2 byte crc16, and 2 bytes per register
        return unpack('>'+('H'*count), resp[3:-2])

    def close(self):
        self.sp.close()

    def send_profile(self, profile):
        ''' profile is a list of 40 tuples (degrees_c, seconds)'''
        base = 0x2000  # base of the profile table
        for p in profile:
            print 'writing profile entry', hex(base)
            print binascii.b2a_hex(self.write_reg(base, p[0]))
            print binascii.b2a_hex(self.write_reg(base+2, p[1]))
            base += 4

    def init_sequence(self):
        print self.read_regs(0x1004, 2)
        print self.read_regs(0x1010, 2)
        print self.read_regs(0x1014, 2)
        print self.read_regs(0x1008, 3)

    def read_profile(self):
        return self.read_regs(0x2000, 80)

    def start(self):
        self.write_reg(0x1018, 1)  

    def stop(self):
        self.write_reg(0x1018, 0)  

    def read_temp(self):
        return self.read_regs(0x1000, 1)[0]

class VirtualTorchOven(object):
    def __init__(self, port = 0):
        self.started = False

    def init_sequence(self):
        pass

    def send_profile(self, profile):
        self.profile = profile

    def read_profile(self):
        return self.profile

    def start(self):
        self.started = True
        self.start_time = time.time()

    def read_temp(self):
        time_decrementor = time.time() - self.start_time
        profile_index = 0
        while time_decrementor >= self.profile[profile_index][1]:
            if profile_index < len(self.profile)-1:
                time_decrementor -= self.profile[profile_index][1]
                profile_index += 1
            else:
                return self.profile[profile_index][0]
                break
        if profile_index == 0:
            last_temp = 15 # Room temperature, let's say
        else:
            last_temp = self.profile[profile_index-1][0]
        return int(last_temp + (time_decrementor/self.profile[profile_index][1])*(self.profile[profile_index][0] - last_temp))

    def stop(self):
        pass

    def close(self):
        pass

if __name__=='__main__':
    from serial.tools.list_ports import comports as comports
    ports = [cp[0] for cp in comports()]
    oven = TorchOven(ports[-1])
    '''
    start temp
    pre-soak 
    '''
    profile = DEFAULT_PROFILE
    try:
        oven.init_sequence()
        oven.send_profile(profile)
        oven.read_profile()
        oven.start()
        try:
            start = time.time()
            while (time.time() - start) < 30:
                try:
                    print oven.read_temp()
                except struct_error as E:
                    print 'Exception reading temp', str(E)
        finally:
            oven.stop()
    finally:
        oven.close()