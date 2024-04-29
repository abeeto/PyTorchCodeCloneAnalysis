bflog = open('bruteforce.log','r')
# Eat header
bflog.readline()

w = 0
mindens = {'wlist':[], 'dist':[], 'time':[], 'dens':[], 'vdet': [], 'isovr': []}

n_time = 1
n_dens = 5
n_dist = 2
n_vdet = 0

for l in bflog:
    ls = l.split()
    mindens['vdet'].append(n_vdet)
    mindens['wlist'].append(w)
    wlog = open('xc12_wn_' + str(w) + '_0001_z1.dat','r')
    # Eat 4 lines as header (3) + junk (1)
    for i in range(0,5):
        wlog.readline()
    t_dens = 0.0
    t_dist = 0.0
    t_time = 0.0
    t_l = wlog.readline()
    t_ls = t_l.split()
    t_time = float(t_ls[n_time])
    t_dist = float(t_ls[n_dist])
    t_dens = float(t_ls[n_dens])
    isovr = False
    for k in wlog:
        ks = k.split()
        k_dens = float(ks[n_dens])
        if (k_dens < t_dens):
            t_dens = k_dens
            t_time = float(ks[n_time])
            t_dist = float(ks[n_dist])
        if (abs(1.0e3 - float(ks[n_time])) < 1.0e-3):
            isovr = True
        
    mindens['dens'].append(t_dens)
    mindens['time'].append(t_time)
    mindens['dist'].append(t_dist)
    mindens['isovr'].append(isovr)
    wlog.close()
    w = w+1
    
# Write out results
mdlogovr = open('mindens_ovr.log','w')
mdlogudr = open('mindens_udr.log','w')

# Write header
mdlogovr.write('w     vdet     time     dist     dens\n')
mdlogudr.write('w     vdet     time     dist     dens\n')
sp = '     '
for j in mindens['wlist']:
    if mindens['isovr'][j]:
        mdlogovr.write(str(j) + sp + str(mindens['vdet'][j]) + sp + 
                       str(mindens['time'][j]) + sp + str(mindens['dist'][j]) + sp + 
                       str(mindens['dens'][j]) + '\n')
    else:
        mdlogudr.write(str(j) + sp + str(mindens['vdet'][j]) + sp + 
                       str(mindens['time'][j]) + sp + str(mindens['dist'][j]) + sp + 
                       str(mindens['dens'][j]) + '\n')
# Close remaining files
bflog.close()    
mdlogudr.close()    
mdlogovr.close()
