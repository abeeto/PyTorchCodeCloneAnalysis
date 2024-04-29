import os

dv = '0.0002e9'

def callgnuplot(j,xc12,vdet):
	global dv	

        os.system('gnuplot -e "inpref=\'xc12_wn_' + j + '_\';' +
                'outfile=\'pres_wn_' + j + '.eps\';' +
                'startxcstr=\'(Initial xc12=' + xc12 + ', vdet=' + vdet + ', dv=' + dv + ')\'" plotP_pm.gp')
        os.system('gnuplot -e "inpref=\'xc12_wn_' + j + '_\';' +
                'outfile=\'dens_wn_' + j + '.eps\';' + 
                'startxcstr=\'(Initial xc12=' + xc12 + ', vdet=' + vdet + ', dv=' + dv + ')\'" plotRho_pm.gp')

	os.system('python getplotmf.py ' + j)
        os.system('gnuplot -e "inpref=\'massfrac_wn_' + j + '.dat\';' + 
                'outfile=\'massfrac_wn_' + j + '.eps\';' + 
                'startxcstr=\'(Initial xc12=' + xc12 + ', vdet=' + vdet + ')\'" plotX.gp')

	os.system('python getplotmf.py ' + j + '_ovr')
        os.system('gnuplot -e "inpref=\'massfrac_wn_' + j + '_ovr.dat\';' +
                'outfile=\'massfrac_wn_' + j + '_ovr.eps\';' +
                'startxcstr=\'(Initial xc12=' + xc12 + ', vdet=' + vdet + ', dv = +' + dv + ')\'" plotX.gp')

	os.system('python getplotmf.py ' + j + '_udr')
        os.system('gnuplot -e "inpref=\'massfrac_wn_' + j + '_udr.dat\';' +
                'outfile=\'massfrac_wn_' + j + '_udr.eps\';' +
                'startxcstr=\'(Initial xc12=' + xc12 + ', vdet=' + vdet + ', dv = -' + dv + ')\'" plotX.gp')
        return

fs = open('scanrange_log.txt','r')

#get header
h = fs.readline()

#plot all scan entries
for l in fs:
	ls = l.split()
	j = ls[0]
	xc12 = '{: > 8,.6F}'.format(float(ls[6]))
	vdet = '{: ^ 18,.16E}'.format(float(ls[2]))
	callgnuplot(j,xc12,vdet)
fs.close()
