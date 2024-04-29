import time
class dab(object):
    def __init__(self, cortesy_time, heat_up_time, dab_time, increment = 1, precision = 0):
        self.inc = increment
        self.prec = precision
        """heat_up_time is how many seconds it will count down for dab_time is how much time it will count up for"""
        print('Get Ready')
        self.timer(cortesy_time)
        print('Heat Up')
        self.timer(heat_up_time)
        print('Cool Down')
        self.timer(dab_time)
        print('DAB TIME!!! CHAIRS!!!')
    def timer(self, n):
        self.fPrint(n)
        iTime = time.time()
        p = 0
        while p < n:
            if time.time() - iTime > p + self.inc:
                p += self.inc
                t = n - p
                self.fPrint(t)
    def fPrint(self, n):
        if n > 59 and n % 60 == 0:
            print('{} min'.format(n // 60))
        elif n > 59:
            print('{} min {:.{}f} sec'.format(n // 60, n % 60, self.prec))
        else:
            print('{:.{}f} sec'.format(n, self.prec))
dab(10, 60, 90, 1, 0)

