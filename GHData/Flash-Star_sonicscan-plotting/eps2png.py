import sys
import os

# Converts all eps files passed on the command line to png

if len(sys.argv)==1:
	exit()

fl = sys.argv[1:]

for f in fl:
	fb = f.rstrip('.eps')
	os.system('ps2pdf -dEPSCrop ' + f + ' ' + fb + '.pdf')
	os.system('gs -o ' + fb + '.png -sDEVICE=png16m  ' + 
			'-dLastPage=1  -r250 ' + fb + '.pdf')
