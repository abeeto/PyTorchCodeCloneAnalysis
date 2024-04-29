import csv
import math


def getdata():
    rates = []
    filename = 'gbp_2001_2018.in'
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        line_count = 0
        for row in csv_reader:
            q = []
            for item in row:
                q.append(item)
            rates.append(q)
            line_count += 1
        ntenors = int(rates[0][0])
        nlines = int(rates[1][0])
        rates.pop(0)
        rates.pop(0)
        rates.pop(0)
    return rates
u =  getdata()
print(u)
