"""
This file is meant to be used on the command line
It is recommended to be used as a command by removing the .py extention, 
making it executable and putting it somewhere your bash interpreter looks
Usage: follow python3 dt.py (or dt) with three, four, or five numerical arguments
These are sent to the constructor in the same order (courtesy, heat up, dab time, etc.)
If no arguments are given or less than three are specified the default is (10, 60, 90, 1, 0)
"""
import time
import sys
class dab(object):
    def __init__(self, cortesy_time, heat_up_time, dab_time, increment = 1, precision = 0):
        self.inc = increment
        self.prec = precision
        print('Get Ready')
        self.timer(cortesy_time, 'Heat Up')
        self.timer(heat_up_time, 'Cool Down')
        self.timer(dab_time, 'DAB TIME!!! CHAIRS!!!')
    def timer(self, n, mess = 0):
        self.fPrint(n)
        iTime = time.time()
        p = 0
        while p < n:
            if time.time() - iTime > p + self.inc:
                p += self.inc
                t = n - p
                if n > p:
                    self.fPrint(t)
                else:
                    print(mess)
    def fPrint(self, n):
        if n > 59 and n % 60 == 0:
            print('{} min'.format(n // 60))
        elif n > 59:
            print('{} min {:.{}f} sec'.format(n // 60, n % 60, self.prec))
        else:
            print('{:.{}f} sec'.format(n, self.prec))

if len(sys.argv) > 5:
    dab(eval(sys.argv[1]), eval(sys.argv[2]), eval(sys.argv[3]), eval(sys.argv[4]), eval(sys.argv[5]))
elif len(sys.argv) > 4:
    dab(eval(sys.argv[1]), eval(sys.argv[2]), eval(sys.argv[3]), eval(sys.argv[4]))
elif len(sys.argv) > 3:
    dab(eval(sys.argv[1]), eval(sys.argv[2]), eval(sys.argv[3]))
else:
    dab(10, 60, 90)
    
