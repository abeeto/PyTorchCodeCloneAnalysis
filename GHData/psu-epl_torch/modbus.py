'''
Created on Mar 21, 2012

@author: Pat Nystrom
Modbus message formatting helper.
'''
import struct
import binascii

class Formatter(object):
    ''' MODBUS polls always have the same essential format regardless
        the polling method. The only differences between serial and 
        TCP polling are the checksum and TCP wrapper. This class knows
        how to format the inner essential part of the poll. It doesn't
        add a checksum or wrapper. All of the methods return a binary
        string beginning with ADDRESS. By default, the address is 1.
        This can be set in the constructor and changed in set_addr.
        
        When a command takes a variable-length parameter, it is a Python
        list of the natural type. In some cases, the list is of tuples.
        See individual command docs.'''
    
    def __init__(self, addr=1):
        self.addr = addr
        
    def set_addr(self, a):
        self.addr = a
    
    # was 'read_coil_status'
    # renamed 9/2/12 for compatibility with modbus spec
    def read_coils(self, start, count):
        ''' start=modbus coil address, 0-relative.
          count = number of coils to read. A coil is one bit.'''
        return struct.pack('>BBHH', self.addr, 1, start, count)
    
    def read_discrete_inputs(self, start, count):    
        ''' start=modbus register address, 0-relative.
          count = number of registers to read. A register is 16-bits.'''
        return struct.pack('>BBHH', self.addr, 2, start, count)
        
    def read_holding_regs(self, start, count):
        ''' start=modbus register address, 0-relative.
          count = number of registers to read. A register is 16-bits.'''
        return struct.pack('>BBHH', self.addr, 3, start, count)

    def read_input_regs(self, start, count):
        ''' start=modbus register address, 0-relative.
          count = number of registers to read. A register is 16-bits.'''
        return struct.pack('>BBHH', self.addr, 4, start, count)

    # was 'force_single_coil'
    # renamed 9/2/12 for compatibility with modbus spec
    def write_single_coil(self, coil, value):
        ''' coil = 0-relative coil (bit) number.
          value = any expression which evaluates as True or False.'''
        return struct.pack('>BBHH', self.addr, 5, coil, 0xff00 if value else 0)
    
    # was 'preset_single_reg'
    # renamed 9/2/12 for compatibility with modbus spec
    def write_single_reg(self, reg, value):
        ''' reg=modbus register address, 0-relative.
          value = 16-bit value to write'''
        return struct.pack('>BBHH', self.addr, 6, reg, value)
    
    def read_exception_status(self):
        return struct.pack('BB', self.addr, 7)
    
    def diagnostics(self, subfunc, data):
        return struct.pack('>BBH', self.addr, 8, subfunc) + data 
    
    def get_comm_event_counter(self):
        return struct.pack('BB', self.addr, 0x0b)
    
    def get_comm_event_log(self):
        return struct.pack('BB', self.addr, 0x0c)
    
    # was 'force_multiple_coils'
    # renamed 9/2/12 for compatibility with modbus spec
    def force_multiple_coils(self, start, values):
        # start is the address of the first coil to set. It is 1-relative; even though
        # internally (in the modbus message) the coil address is zero-relative, the 
        # documentation for a modbus device will typically call the lowest coil 1.
        # values is an iterable of values which are evaluated as booleans.
        vstr = ''   # value string - the list of bytes containing the coil bits
        vbyte = 0   # byte being assembled
        nbits = 0   # how many bits in the assembled byte, so far
        for v in values:
            vbyte <<= 1
            if v:
                vbyte |= 1
            nbits += 1
            if nbits == 8:
                vstr += chr(vbyte)  # append completed 8-bits
                vbyte = 0
                nbits = 0
        if nbits:
            vstr += chr(vbyte)  # append partially-assembled last byte
        return struct.pack('>BBHHB', self.addr, 0x0f, start, len(values), len(vstr))+vstr
    
    # was 'preset_multiple_regs'
    # renamed 9/2/12 for compatibility with modbus spec
    def write_multiple_regs(self, start, regs):        
        # regs is an iterable of 16-bit values
        return struct.pack('>BBHHB'+('H'*len(regs)), self.addr, 0x10, start, len(regs), len(regs)*2, *regs)

    def report_slave_id(self):
        return struct.pack('BB', self.addr, 0x11)
    
    def read_file_record(self, reqs):
        ''' reqs is a list of tuples (file_num, start_rec, rec_len) '''
        hdr = struct.pack('>BBB', self.addr, 0x14, 7*len(reqs))
        for r in reqs:
            hdr += struct.pack('>BHHH', 6, r[0], r[1], r[2])
        return hdr
     
    def write_file_record(self, reqs):
        ''' reqs is a list of tuples (file_num, rec_num, data) 
          data is a list of integer values to write to the file. Each entry of
          data will turn into 16 bits in the message.'''
        data_part = ''
        for r in reqs:
            wordlen = len(r[2])    # how many 16-bit words are passed in to convert  
            fmt = '>BHHH' + ('H'*wordlen)   # 3 half-words of header + 1 half-word for each datum
            data_part += struct.pack(fmt, 6, r[0], r[1], len(r[2]), *(r[2]))  # *(r[2]) unpacks the data list as parameters to pack
        hdr = struct.pack('>BBB', self.addr, 0x15, len(data_part))
        return hdr + data_part
    
    def mask_write_register(self, addr, _and, _or):
        return struct.pack('>BBHHH', self.addr, 0x16, addr, _and, _or)
    
    # was 'rw_4x_regs'
    # renamed 9/2/12 for compatibility with modbus spec
    def rw_multiple_regs(self, read_start, read_count, write_start, regs):
        return struct.pack('>BBHHHHB'+('H'*len(regs)), self.addr, 0x17, read_start,
                           read_count, write_start, len(regs), len(regs)*2, *regs)
        
    def read_fifo(self, addr):
        return struct.pack('>BBH', self.addr, 0x18, addr)
#    def mb_enter_mon(self):    
#        # 'enter monitor' is modbus command register code 0x7000
#        self.send_mb_command(0x7000)


        
if __name__=='__main__':
    from binascii import b2a_hex
#    from mux import TMux
    import socket
    import time
    import threading
    
#    m = TMux(nchan=8)
#    try:
#        m.analog_off()
#        m.set_baud_rate(1)
#    finally:
#        m.cleanup()

    class TestThread(threading.Thread):
        def __init__(self):
            super(TestThread, self).__init__()
            self.die = False
            self.f = Formatter()

        def connect(self, addr='192.168.1.80'):
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(1)
            self.sock.connect((addr, 502))
            self.sock.settimeout(0.25)
            
        def send_recv(self, msg):    
            m = struct.pack('>H3sB', 1, '', len(msg))+msg  # TID, 3 zeroes, length to follow, then message
            print 'sending', b2a_hex(m)
            self.sock.send(m)
            try:
                d = self.sock.recv(100)
                print 'got', b2a_hex(d)
                return d
            except socket.timeout:
                print 'timeout'
                return None

        def run(self):
            conn_count = 1
            while not self.die:
                try:
                    if self.send_recv(self.f.read_holding_regs(0, 2)):
                        self.sock.close()
                        self.connect()
                        conn_count += 1
                        print 'conn_count =', conn_count
                except Exception as E:
                    print 'Exception in run', str(E)
        
        def kill(self):
            print 'killing'
            self.die = True
            self.join(2)
            if self.sock:
                self.sock.close()
            print 'killed'
            
    class TelTestThread(threading.Thread):
        def __init__(self):
            super(TelTestThread, self).__init__()
            self.die = False

        def connect(self, addr='192.168.1.80'):
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(1)
            self.sock.connect((addr, 23))
            self.sock.settimeout(0.25)
            
        def run(self):
            conn_count = 1
            while not self.die:
                try:
                    d = self.sock.recv(2048)
                    if d:
                        print d
                    else:
                        print 'did not get data'
                    self.sock.close()
                    self.connect()
                    conn_count += 1
                    print 'conn_count =', conn_count
                except Exception as E:
                    print 'Exception in run', str(E)
        
        def kill(self):
            print 'killing'
            self.die = True
            self.join(2)
            if self.sock:
                self.sock.close()
            print 'killed'
            
    class SerTestThread(threading.Thread):
        def __init__(self, port):
            super(SerTestThread, self).__init__()
            self.port = port
            self.die = False
        
        def run(self):
            while not self.die:
                c = self.port.read(1)
                if c:
                    numch = self.port.inWaiting()
                    if numch:
                        c += self.port.read(numch)
                    print binascii.b2a_hex(c)
                
        def kill(self):
            self.die = True
            self.join(2)
            print 'killed'

    import serial
    from ne.basf_ext.basf_ext import modbus_crc16

    f = Formatter()
    print binascii.b2a_hex(f.write_file_record([
        (4, 7, [0x6af, 0x4be, 0x100d])
                                                ]))
#    P = serial.Serial('COM40', 115200)                
#    try:
#        P.setTimeout(0.25)
#        T = SerTestThread(P)                
#        T.start()
#        mb = Formatter()
#        while True:
#            s = raw_input()
#            if not s.strip():
#                break
#            c = mb.read_holding_regs(0, 2)
#            P.write(c + modbus_crc16(c))
#        T.kill()
#    finally:
#        P.close()
        
#    
#    try:
#        raw_input()
#    finally:
#        send_recv(f.preset_multiple_regs(0x80, [0, 2, 4, 6]))
#        for i in xrange(10):
##            send_recv(f.preset_multiple_regs(0x80, [1]))
##            send_recv(f.force_multiple_coils(1, [True]*2))
##            send_recv(f.read_coil_status(1, 3))
##            send_recv(f.read_exception_status())
#            d = send_recv(f.read_holding_regs(0x80, 4))
#            print struct.unpack('>ff', d[-8:])
##            send_recv(f.preset_single_reg(0x80, 0x1234))
##            send_recv(f.force_single_coil(1, True))
##        send_recv(f.rw_4x_regs(1, 5, 0x123, [1,2,3]))
##        send_recv(f.read_fifo(100))
#        
##        
##        print b2a_hex(f.preset_multiple_regs(0x500, [1,2,3,4,5]))    
##        print b2a_hex(f.force_multiple_coils(1, [True]*2))
##        print b2a_hex(f.read_coil_status(1, 3))
##        print b2a_hex(f.read_exception_status())
##        print b2a_hex(f.read_holding_regs(1, 10))
##        print b2a_hex(f.preset_single_reg(1, 0x1234))
##        print b2a_hex(f.force_single_coil(1, True))
##        print b2a_hex(f.rw_4x_regs(1, 5, 0x123, [1,2,3]))
##        print b2a_hex(f.read_fifo(100))
#    finally:
#        sock.close()