import glob
import os

fl = glob.glob('*.eps')

s = 'python eps2png.py '
for f in fl:
	s = s + f + ' '
os.system(s)
