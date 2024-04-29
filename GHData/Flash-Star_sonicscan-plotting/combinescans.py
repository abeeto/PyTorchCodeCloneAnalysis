import glob

# expects scan logs to be of the form 'scanrange_log_#.txt' where # is the ordering
# output filename is scanrange_log.txt
fon = 'scanrange_log.txt'
fl = glob.glob('scanrange_log_*.txt')

files = {}

# ordering number must start with 1 and increment with each successive file

flargest = 1
for f in fl:
	fs = f.rstrip('.txt').split('_')
	fn = int(fs[2])
	if (fn > flargest):
		flargest=fn
	files[str(fn)] = open(f,'r')

# append files...

fo = open(fon,'w')

for i in range(1,flargest+1):
	si = str(i)
	if (i==1):
		# Write header only for the first file
		for l in files[si]:
			fo.write(l)
	else:
		files[si].readline()
		for l in files[si]:
			fo.write(l)

# close everything
fo.close()
for k,f in files.iteritems():
	f.close()
			



