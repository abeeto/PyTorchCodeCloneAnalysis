import os

# Convert eps files to png
# os.system('python eps2pngall.py')

inpre_P = 'pres_wn_'
inpre_Rho = 'dens_wn_'
inpre_X = 'massfrac_wn_'

otpre_P = 'pres'
otpre_Rho = 'dens'
otpre_X = 'massfrac'

cmd1 = 'ffmpeg -r 2 -y -start_number 0 -i '
cmd2 = '.png -c:v libx264 -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -r 30 '
cmd3 = '.mp4'

cmd = cmd1 + inpre_P + '%d' + cmd2 + otpre_P + cmd3
os.system(cmd)

cmd = cmd1 + inpre_Rho + '%d' + cmd2 + otpre_Rho + cmd3
os.system(cmd)

cmd = cmd1 + inpre_X + '%d' + cmd2 + otpre_X + cmd3
os.system(cmd)
