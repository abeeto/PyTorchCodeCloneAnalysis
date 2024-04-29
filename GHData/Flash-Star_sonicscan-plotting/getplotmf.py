from collections import OrderedDict
import sys

# pass this program one integer on the command line to tell it which wn to use.
# if no integer is passed, assume using none.

wnstr = ''
outfilename = ''
if len(sys.argv) == 1:
	print 'Neglecting wn...'
	outfilename = 'massfrac.dat'
else:
	wn = sys.argv[1]
	wnstr = wn + '_'
	outfilename = 'massfrac_wn_' + wn + '.dat'

prefix = 'xc12_wn_' + wnstr
suffix = '_z1.dat'

fout = open(outfilename,'w')

# These numbers are the integer before _z1.dat in the torch output files
# # where these elemental compositions are located.
fnums = OrderedDict([('c12',3),('o16',4),('ne20',5),('ne22',5),('si28',8),
			('ca40',12),('ni56',24),('fe54',21)])
# These numbers are the column number for the respective element compositions
fcols = OrderedDict([('c12',6),('o16',6),('ne20',7),('ne22',9),('si28',2),
                        ('ca40',6),('ni56',4),('fe54',9)])
# These are storage locations for the mass fractions from the input files
massf = OrderedDict([('c12',[]),('o16',[]),('ne20',[]),('ne22',[]),('si28',[]),
                        ('ca40',[]),('ni56',[]),('fe54',[])])
# These are the storage locations for the thermo variables
thermo = OrderedDict([('time',[]),('dist',[]),('vel',[]),('temp',[]),
			('dens',[]),('pres',[]),('ener',[])])
# These numbers are the column number for the thermo variables
thermocols = OrderedDict([('time',1),('dist',2),('vel',3),('temp',4),
                        ('dens',5),('pres',6),('ener',7)])
variables = OrderedDict()
files = OrderedDict()
numpts = 0

thermof = open(prefix + '0001' + suffix,'r')

# Open files in fnums
for k,v in fnums.iteritems():
	if not str(v) in files:
		files[str(v)] = open(prefix + '{:0>4}'.format(v) + suffix,'r')
		
# Eat headers
for k,v in files.iteritems():
	for i in range(0,3): v.readline()
## Eat an extra line in thermo file to get real start point
for i in range(0,4): thermof.readline()

# Fill dicts with data
for l in thermof:
	ls = l.split()
	for k,v in thermo.iteritems():
		thermo[k].append(ls[thermocols[k]])
	numpts = numpts + 1

for k,v in files.iteritems():
	for l in v:
		ls = l.split()
		for kk,vv in fnums.iteritems():
			if str(vv) == k:
				massf[kk].append(ls[fcols[kk]])
# Print some output...
#print 'numpts: ' + str(numpts)
#print 'length of data lists:'
#for k,v in thermo.iteritems():
#	print k + ': ' + str(len(v))
#for k,v in massf.iteritems():
#	print k + ': ' + str(len(v))

# Close input files
thermof.close()
for k,v in files.iteritems():
	v.close()

# Join dictionaries
for k,v in thermo.iteritems():
	variables[k] = v
for k,v in massf.iteritems():
	variables[k] = v

# Write to output file
## Header
for k,v in variables.iteritems():
	fout.write(k + '          ')
fout.write('\n')
## Data
for j in range(0,numpts):
	for k,v in variables.iteritems():
		fout.write(v[j] + ' ')
	fout.write('\n')

# Close output file
fout.close()
	



