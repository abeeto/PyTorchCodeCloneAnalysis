fsr = open('scanrange_log.txt','r')

#Eat header
fsr.readline()

#Check all compositions for norm
for l in fsr:
	ls = l.split()
	x = ls[3:]
	xnorm = 0.0
	for xj in x:
		xnorm = xnorm + float(xj)
	print 'wn: ' + ls[0] + ', norm = ' + str(xnorm)

fsr.close()	
