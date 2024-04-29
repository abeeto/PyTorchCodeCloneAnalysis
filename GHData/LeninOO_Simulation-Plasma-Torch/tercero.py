

import numpy as np
import matplotlib.pyplot as plt
import random
import math
from math import log
from libreria.operadores import *
from libreria.inicio import *
import datetime


########################  1.- INICIO  DEFINICION DE CONSTANTES #################################################################################
f = 10   #  Resolucion de la malla
#--------------------------------- INICIO Cargo  puntos de la malla -----------------------------------------------------------------------------------
corde = np.array([[[0.0001,0.0001] for col in range(f+1)] for row in range(f+1)])
archivo = open("MALLA 1/Final_Grid.txt","r")
lineas = archivo.readlines()
t= 0
for w in range(0,f+1):
       
       for z in range(0,f+1):
         punto =lineas[w*f+z+t+6]
         punto = punto.split()
         punto[0] = punto[0].replace(",",".")
         punto[1] = punto[1].replace(",",".")
         corde[w][z] = punto
         
       t=t+1
archivo.close()
#                                   corde[Y][X]     
#--------------------------------- FIN  cargar  los puntos de la malla ----------------------------------------------------------------------------------

d_tiempo_all = np.array([[0.000000  for col in range(f+1)] for row in range(f+1)])

U1 = np.array([[0.00001  for col in range(f+1)] for row in range(f+1)])
U2 = np.array([[0.00001  for col in range(f+1)] for row in range(f+1)])
U3 = np.array([[0.00001  for col in range(f+1)] for row in range(f+1)])
U4 = np.array([[0.00001  for col in range(f+1)] for row in range(f+1)])
U5 = np.array([[0.00001  for col in range(f+1)] for row in range(f+1)])
U6 = np.array([[0.00001  for col in range(f+1)] for row in range(f+1)])

U1_temp = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
U2_temp = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
U3_temp = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
U4_temp = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
U5_temp = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
U6_temp = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])

diff_n_U1 = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
diff_n_U2 = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
diff_n_U3 = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
diff_n_U4 = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
diff_n_U5 = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
diff_n_U6 = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])

diff_np1_U1 = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
diff_np1_U2 = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
diff_np1_U3 = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
diff_np1_U4 = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
diff_np1_U5 = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
diff_np1_U6 = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])

F1 = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
F2 = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
F3 = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
F4 = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
F5 = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
F6 = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])

G1 = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
G2 = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
G3 = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
G4 = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
G5 = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
G6 = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])

D1 = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
D2 = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
D3 = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
D4 = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
D5 = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
D6 = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])

S1 = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
S2 = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
S3 = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
S4 = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
S5 = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
S6 = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])

#----------------------------  Variables  Inicializacion ---------------------------

Tg = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
Te = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
u = np.array([[0.00001  for col in range(f+1)] for row in range(f+1)])
v = np.array([[0.00001  for col in range(f+1)] for row in range(f+1)])

u_e = np.array([[0.00001  for col in range(f+1)] for row in range(f+1)])
v_e = np.array([[0.00001  for col in range(f+1)] for row in range(f+1)])


densidad_gas = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
densidad_electron = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
presion = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
B = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
E = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
o_= np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
Energia_gas = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
Energia_electron = np.array([[0.00001  for col in range(f+1)] for row in range(f+1)])
ne_hh = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])

#--------------------------------------------------------------------------------------
#------------------------------- Variables  auxiliares ---------------------------------
n_n_p  = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
n_e_p  = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
n_n_c  = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
n_e_c  = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])

ne_hh_p  = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
ne_hh_c  = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])

Qnn  = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
Qen  = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
Qee  = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
Qei  = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
Qii  = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
V  = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
vis  = np.array([[0.01  for col in range(f+1)] for row in range(f+1)])
Da  = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
fre_ei  = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
fre_en  = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
El  = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
Ce  = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
Cg  = np.array([[0.01  for col in range(f+1)] for row in range(f+1)])
R   = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
Qen3100 = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
S = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
Ke = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
Kg = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
q_gx = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
q_gy = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
Hg  = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
estres_xx = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
estres_xy = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
estres_yy = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
q_ex = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
q_ey = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
jx = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
jy = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
c_factor = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
gradiente_campo_E = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
densidad_carga = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
carga = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
densidad_corriente = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
tenedor = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
gradiente_presion_electron = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])
intensidad_corriente = np.array([[0.0001  for col in range(f+1)] for row in range(f+1)])


#----------------------------------------------------------------------------------------

U1_completo=[]
U2_completo=[]
U3_completo=[]
U4_completo=[]
U5_completo=[]
U6_completo=[]

F1_completo=[]
F2_completo=[]
F3_completo=[]
F4_completo=[]
F5_completo=[]
F6_completo=[]

G1_completo=[]
G2_completo=[]
G3_completo=[]
G4_completo=[]
G5_completo=[]
G6_completo=[]


S1_completo=[]
S2_completo=[]
S3_completo=[]
S4_completo=[]
S5_completo=[]
S6_completo=[]

#---------------------------------------  Llamo a la inicializacion ------------------------------------------------------------------
Io = 800
Radio_c = 5e-3
ini_temperatura_gas( Tg,300,f )
ini_temperatura_electron(Te,800,f)

ini_velocidad_v(v,f)
ini_velocidad_u(u,f)
ini_velocidad_v_e(v_e,f)
ini_velocidad_u_e(u_e,f)

ini_densidad_gas(densidad_gas,f)
ini_densidad_electron(densidad_electron,f)
ini_campo_magnetico(B,Io,Radio_c,f)
ini_campo_electrico(E,f)


tempo_300= ini_potencial(f,f,1000)
for i in range(0,10):
                 for j in range(0,10):
                              E[i+1][j+1]= tempo_300[i][j]
ini_Energia_gas(Energia_gas,f)
ini_Energia_electron(Energia_electron,f)

# Necesito definir los 24  coeficientes para los suavisadores  

unod4x, unod4y, unod2x, unod2y = 0.1 , 0.1, 0.1, 0.1
dosd4x, dosd4y, dosd2x, dosd2y = 0.1 , 0.1, 0.1, 0.1
tresd4x, tresd4y, tresd2x, tresd2y = 0.1 , 0.1, 0.1, 0.1
cuatrod4x, cuatrod4y, cuatrod2x, cuatrod2y = 0.1, 0.1, 0.1, 0.1
cincod4x,  cincod4y, cincod2x, cincod2y = 0.1, 0.1, 0.1, 0.1
seisd4x,  seisd4y, seisd2x, seisd2y = 0.1, 0.1, 0.1, 0.1


pi = 3.14159265359
u_o = 4*pi*10**(-7)                  # Permeabilidad en el vacio H/m
e_o = 8.854*10**(-12)                # permitividad en el vacio F/m
e  = 1.602*10**(-19)                 # Electric charge of a proton Colulombs       
h  = 6.626*10**(-34)                 # Constante de Planck  J - s
k  = 1.381*10**(-23)                 # Constante de Botzman J/K
me = 9.11*10**(-31)                  # masa del electron kg
ma = 6.634*10**(-26)                 # masa atomica del argon kg
Ei = 2.53*10**(-18)                  # Energia de ionizacion para el argon en J, primera ionizacion
calor_e = 1.667                      # Ratio of specific heats for Argon  y = 1.667
Ra = 208.13                          #  Constante de gas para el argon J/(kgK)
Re = 1.516e7#7                   # Electron gas constant  J/kgK           
Rh = 8249.23                         # Constante de gas para el hidrogeno J/kgK  
I = 3000                             # Corriente inicial
Radio_c = 0.005                      # Radio del catodo
P_o =  1                             # Presion inicial en atmosferas
Sk  =  0.1                           # Factor de correccion de ploteo
So  =  0.1                           # Factor de correccion polinomio de Sonine
ceA = 1.667                          # Calor especifico para el Argon
factor_seguridad_tiempo=0.5          # Coeficiente  de seguridad para el paso de tiempo
Qin = 1.4*10**(-18)
error = 0.5
vuelta_electrones = 0
dt_electrones = 0


#-------------------------------------- 2   Definir   todas las U  en base a las variables libres  ------------------------------------#
U1= U1_temp = densidad_gas 
U2= U2_temp = densidad_electron 
U3= U3_temp = U1*u 
U4= U4_temp = U1*v 
U5= U5_temp = U1*Energia_gas 
U6= U6_temp = U2*Energia_electron 
#--------------------------------------------------------------------------------------------------------------------------------------#
#--------------------------------------  3  Redefinir   las variables libres en base a  U  despejando ---------------------------------#
u = U3/U1
v = U4/U1

Energia_gas = U5/U1
Energia_electron = U6/U1

#--------------------------------------------------------------------------------------------------------------------------------------#

######################   3.-INICIO     BUCLE PRINCIPAL ##############################
ct = datetime.datetime.now()
for time in range(1):
   
#while error > 0.1  :

###########################################  INICIO DEL PREDICTOR ESPECIES PESADAS #############################################################
    
 #--------------------------------------------  1  Definir los  terminos  F G  S en base  a  U -------------------------------------------------#
    print( " Esta es la vuelta GAS IONES+MOLECULAS ",time)
    
    #--------------------------- Calculando  B  --------------------------------------------------------------------------------------------------------------------#
    delta_x = delta( "x", corde,f )
    delta_y = delta( "y", corde,f )
    Ce =  (abs(k*Te/me))**(1/2)
    Cg = (abs(k*Tg/ma))**(1/2)             # Velocidad termica del gas   
    Cg1= 3**(1/2)*Cg
    n_n_p = (U1*Ra)/k                                                #  densidad de las particulas neutrales
    n_e_p = (U2*Re)/k   
    Hg = U5/U1 +Ra*Tg 
    presion = U1*Ra*Tg + U2*Re*Te     # presion  integral
    
    V = 1.239e7*(abs(Te/n_e_p))**(1/2)
    for j in range(1,f+1):
                      for i in range(1,f+1):
                              Qen3100[j][i] = (-0.488+3.96*10**(-4)*Te[j][i])*10**(-20)                  # "Qen3100"  si  la temperatura es mayor a 3000  grados
                              Qei[j][i] =  ((e**4)*log(V[j][i]))/24*pi*(e_o*k*Te[j][i])**2                #  "e_o" permitividad en el vacio , "Te" Temperatura del electron, "V" factor sparta, "k" Constante de Botzman
                              Qin = 1.4*10**(-18)                                            #  Seccion de colision entre ion y neutron en metros cuadrados
                              Qnn[j][i] = 1.7*10**(-18)*Tg[j][i]**(1/4)                                  #  Seccion de colision cruzada entre neutron y neutron
                              Qen[j][i] = (0.713 - 4.5*10**(-4)*Te[j][i] + 1.5*10**(-7)*Te[j][i]**2)*10**(-20) #  Seccion de colision cruzada de electron y neutron
                              Qee[j][i] =  ((e**4)*log(V[j][i]))/24*pi*(e_o*k*Te[j][i])**2
                              Qii[j][i] = ((e**4)*log(V[j][i]))/24*pi*(e_o*k*Tg[j][i])**2                      # "V" factor Sparta, "e_o" permitividad en el vacio, "Tg" temperatura gas,"k" Constante de Boltzman
                   
    tenedor = e/(me*Ce*(n_e_p*Qei + n_n_p*Qen3100))
    Pe = U2*Te*Re
    diff_Pex =  diff_predictor(Pe ,"x", corde,f ) #  Diferencial del campo "B" Magnetico en la coordenada  y
    diff_Pey =  diff_predictor(Pe ,"y", corde,f ) #  Diferencial del campo "B" Magnetico en la coordenada  y
    gradiente_presion_electron = diff_Pex +diff_Pey
    o_ = 0.1*(e**2*n_e_p)/(me*Ce*(n_e_p*Qei + n_n_p*Qen3100))
    densidad_corriente = E*o_ + tenedor*gradiente_presion_electron
    for j in range(1,f+1):
                      for i in range(1,f+1):
                                      intensidad_corriente[j][i] = densidad_corriente[j][i]*delta_x[j][i]*delta_y[j][i] 
                                      B[j][i]=  u_o*intensidad_corriente[j][i]/2*pi*delta_y[j][i]
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    
    diff_xx_x =  diff_predictor(u,"x", corde,f )
    diff_xx_y =  diff_predictor(v ,"y", corde,f )
    diff_xy_x =  diff_predictor(v , "x", corde,f )
    diff_xy_y =  diff_predictor(u , "y", corde,f )
    diff_yy_x =  diff_predictor(u , "x", corde,f )
    diff_yy_y =  diff_predictor(v , "y", corde,f )
    diff_gx =  diff_predictor(Tg , "x", corde,f ) #  "Tg" Temperatura de gas, "corde" toda las coordenadas de la malla 
    diff_gy =  diff_predictor(Tg , "y", corde,f )
    diff_Bx =  diff_predictor(B ,"x", corde,f ) #  Diferencial del campo "B"  Magnetico en la coordenada x
    diff_By =  diff_predictor(B ,"y", corde,f ) #  Diferencial del campo "B" Magnetico en la coordenada  y
                  
    
    vis = ((0.5*ma*Cg)/(2**(1/2)))*(n_n_p/(n_n_p*Qnn + n_e_p*Qin) + n_e_p/(n_n_p*Qin + n_e_p*Qii)) #  calculo de la viscosidad            
    estres_xx = (4/3)*vis*(diff_xx_x) - (2/3)*vis*(diff_yy_y) 
    estres_xy = vis*(diff_xy_y) + vis*(diff_xy_x) 
    estres_yy = (4/3)*vis*(diff_yy_y) - (2/3)*vis*(diff_yy_x) 
    El = ((3*U2)/ma)*(n_e_p*Ce*Qei + n_e_p*Ce*Qen)*k*(Te - Tg) 
    Kg = Sk*(k**2*Tg/ma*Cg)*( n_n_p/(n_n_p*Qnn + n_e_p*Qin)+ n_e_p/(n_e_p*Qii +n_n_p*Qin)) # "Kg" factor de transmision del calor del gas , "Cg"  velocidad termica del gas
    o_ = So*(e**2*n_e_p)/(me*Ce*(n_e_p*Qei + n_n_p*Qen))# "o_ " conductividad electrica  1.975 , "So"  coeficiente Sonine , "Ce" velocidad  termica del electron ,"n_e" densidad del electron, "n_n" densidad neutron     
    q_gx = -Kg*(diff_gx) #   "q_gx"  Difusion del calor del gas en la coordenada x
    q_gy = -Kg*(diff_gy) #   "q_gy"  Difusion del calor del gas en la coordenada y
    
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    # 1.-presion   : presion =U1*Ra*Tg + U2*Re*Te  :   Tg,Te
    # 2.-B         : B = (u_o*I)/2*pi*Radio_c      :   ???????
    # 3.-estres_xx : (4/3)*vis[j][i]*(diff_xx_x[j][i]) - (2/3)*vis[j][i]*(diff_yy_y[j][i])  :  vis, diff_xx_x ,  dif_yy_y
    # 4.-estres_xy :  vis[j][i]*(diff_xy_y[j][i]) + vis[j][i]*(diff_xy_x[j][i])             :  vis , diff_xy_y , diff_xy_x
    # 5.- Hg       : Energia_gas[j][i] +Ra*Tg[j][i] :  Energia_gas, Tg
    # 6.- q_gx     : -Kg[j][i]*(diff_gx[j][i])     :  Kg,diff_gx 
    # 7.- estres_yy: (4/3)*vis[j][i]*(diff_yy_y[j][i]) - (2/3)*vis[j][i]*(diff_yy_x[j][i])  :  vis , diff_yy_y , diff_yy_x
    # 8.- q_gy     :  -Kg[j][i]*(diff_gy[j][i])    :  Kg ,diff_gy
    # 9.- El       : ((3*densidad_electron[j][i])/ma)*(su_fre)*k*(Te[j][i] -Tg[j][i]) :  densidad_electron ,su_fre ,Te , Tg
    
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    
    
    F1 = U3
    F3 = (U3*U3)/U1 +presion + (B*B)/2*u_o  - estres_xx  # "u_o" Permeabilidad en el vacio H/m
    F4 = (U3*U4)/U1 - estres_xy
    F5 = U3*Hg -U3/U1*estres_xx - U4/U1*estres_xy + q_gx # "q_gx"  difusion del calor del gas en la coordenada x
    
    G1 = U4
    G3 = (U3*U4)/U1 - estres_xy
    G4 = (U4*U4)/U1 + presion + (B*B)/2*u_o - estres_yy 
    G5 = U4*Hg - U3/U1*estres_xy - U4/U1*estres_yy + q_gy #  "q_gy" difusion del calor del gas en la coordenada y
    
    S1 = 0
    S3 = 0
    S4 = 0                                                
    S5 = El                                               #   "El" energia de transferencia colisionale entre los electrones y las particulas pesadas
    
   #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
   
   
   #-------------------  2   Derivativo  predictivo de  F G  S    PREDICTOR ----------------------------------------------------------------------------------------------------------------------------#
    
    F1_predictor = diff_predictor(F1 ,"x", corde,f ) 
    F3_predictor = diff_predictor(F3 ,"x", corde,f )
    F4_predictor = diff_predictor(F4 ,"x", corde,f )
    F5_predictor = diff_predictor(F5 ,"x", corde,f )
    
    G1_predictor = diff_predictor(G1 ,"y", corde,f )
    G3_predictor = diff_predictor(G3 ,"y", corde,f )
    G4_predictor = diff_predictor(G4 ,"y", corde,f )
    G5_predictor = diff_predictor(G5 ,"y", corde,f )
    
        
    diff_n_U1 = S1 - F1_predictor -  G1_predictor
    diff_n_U3 = S3 - F3_predictor -  G3_predictor
    diff_n_U4 = S4 - F4_predictor -  G4_predictor
    diff_n_U5 = S5 - F5_predictor -  G5_predictor
    
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    
    #-----------------------------  3   Calcular el time  y el  suavizador ------------------------------------------------------------------------------------------------------------------------------#
    
    ################### Delta  de Tiempo   ESPECIES  PESADAS ############################################################################
    
    d_tiempo_all = np.array([[0.0001  for col in range(f)] for row in range(f)])
     
    
    delta_x = delta( "x", corde,f )
    delta_y = delta( "y", corde,f )
    
    for i in range(1,f+1):    
      for w in range(1,f+1):
         c_factor = ( abs(ceA*Ra*Tg[i][w])  )**(1/2)
         Re_delta_x = (U1[i][w]*abs(u[i][w])*delta_x[i][w])/vis[i][w]
         Re_delta_y = (U1[i][w]*abs(v[i][w])*delta_y[i][w])/vis[i][w]
         Re_delta = min( Re_delta_x, Re_delta_y)
         delta_t_CFL = (u[i][w]/delta_x[i][w] + v[i][w]/delta_y[i][w] + c_factor*(1/(delta_x[i][w])**2+ 1/delta_y[i][w]**2)**(1/2) )**(-1)
         delta_tiempo = (factor_seguridad_tiempo*delta_t_CFL)/(1+ 2/Re_delta)
         d_tiempo_all[i-1][w-1] = delta_tiempo 
         dt = np.amin(d_tiempo_all)
    print(" Tiempo  gases ",dt) 
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    
    #-----------------------------  4    Definir  el U_temp  -------------------------------------------------------------------------------------------------------------------------------------------#
    U1_temp=U1+dt*diff_n_U1
    U3_temp=U3+dt*diff_n_U3
    U4_temp=U4+dt*diff_n_U4
    U5_temp=U5+dt*diff_n_U5
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    
    #------------------------------- 5   CONDICIONES  FRONTERA ESPECIES PESADAS  1 ---------------------------------------------------------------------------------------------------------------------#
    
    #-------------------------------LINEA  0 ---------------------------#
    for q in range(f):
        if 1 <= i <= 25:
            U1_temp[0][q]= 0.01
            U3_temp[0][q]= 0.01
            U4_temp[0][q]= 0.01
            U5_temp[0][q]= 0.01
            
        elif 25 < i <= 29:
            U1_temp[0][q]= 0.1
            U3_temp[0][q]= 0.10
            U4_temp[0][q]= 0.10
            U5_temp[0][q]= 0.10
           
        elif 29 < i <= 60:
            U1_temp[0][q]= 0.1
            U3_temp[0][q]= 0.1
            U4_temp[0][q]= 0.1
            U5_temp[0][q]= 0.1
            
        else:
            U1_temp[0][q]= 0.1
            U3_temp[0][q]= 0.1
            U4_temp[0][q]= 0.1
            U5_temp[0][q]= 0.1
    
    #-------------------------------------------------------------------#
    #----------------LINEA   FINAL -------------------------------------# 
    for r in range(f):
            if  0 <= i <= 15:
                U1_temp[f][r]= 0.1
                U3_temp[f][r]= 0.1
                U4_temp[f][r]= 0.1
                U5_temp[f][r]= 0.1
                
            elif 15 < i <= 32:
                U1_temp[f][r]= 0.1
                U3_temp[f][r]= 0.1
                U4_temp[f][r]= 0.1
                U5_temp[f][r]= 0.1
               
            elif 32 < j <= 60:
                U1_temp[f][r]= 0.1
                U3_temp[f][r]= 0.1
                U4_temp[f][r]= 0.1
                U5_temp[f][r]= 0.1
                
            else:
                U1_temp[f][r]= 0.1
                U3_temp[f][r]= 0.1
                U4_temp[f][r]= 0.1
                U5_temp[f][r]= 0.1
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    
    

    #------------------------------------------------------------- 6    Definir las variables libre en funcion a  U_temp -------------------------------------------------------------------#
   
    u_temp = U3_temp/U1_temp #  es  "u"
    v_temp = U4_temp/U1_temp #  es  "v"
    Tg_temp = 0.0032031*(U5_temp/U1_temp - 0.5*(U3_temp*U3_temp/U1_temp*U1_temp - U4_temp*U4_temp/U1_temp*U1_temp))
    Tg_temp = abs(Tg_temp)
    
    Ce =  (abs(k*Te/me))**(1/2)
    Cg = (abs(k*Tg_temp/ma))**(1/2)             # Velocidad termica del gas   
    Cg1= 3**(1/2)*Cg
    n_n_c = (U1_temp*Ra)/k                                                #  densidad de las particulas neutrales
    n_e_c = (U2*Re)/k   
    Hg = U5_temp/U1_temp +Ra*Tg_temp 
    presion = U1_temp*Ra*Tg_temp + U2*Re*Te     # presion  integral
        
    V = 1.239e7*(abs(Te/n_e_c))**(1/2)
    for j in range(1,f+1):
                     for i in range(1,f+1):
                                  Qen3100[j][i] = (-0.488+3.96*10**(-4)*Te[j][i])*10**(-20)                  # "Qen3100"  si  la temperatura es mayor a 3000  grados
                                  Qei[j][i] =  ((e**4)*log(V[j][i]))/24*pi*(e_o*k*Te[j][i])**2                #  "e_o" permitividad en el vacio , "Te" Temperatura del electron, "V" factor sparta, "k" Constante de Botzman
                                  Qin = 1.4*10**(-18)                                            #  Seccion de colision entre ion y neutron en metros cuadrados
                                  Qnn[j][i] = 1.7*10**(-18)*Tg_temp[j][i]**(1/4)                                  #  Seccion de colision cruzada entre neutron y neutron
                                  Qen[j][i] = (0.713 - 4.5*10**(-4)*Te[j][i] + 1.5*10**(-7)*Te[j][i]**2)*10**(-20) #  Seccion de colision cruzada de electron y neutron
                                  Qee[j][i] =  ((e**4)*log(V[j][i]))/24*pi*(e_o*k*Te[j][i])**2
                                  Qii[j][i] = ((e**4)*log(V[j][i]))/24*pi*(e_o*k*Tg_temp[j][i])**2                      # "V" factor Sparta, "e_o" permitividad en el vacio, "Tg" temperatura gas,"k" Constante de Boltzman
                       
    tenedor = e/(me*Ce*(n_e_c*Qei + n_n_c*Qen3100))
    Pe = U2*Te*Re
    diff_Pex =  diff_predictor(Pe ,"x", corde,f ) #  Diferencial del campo "B" Magnetico en la coordenada  y
    diff_Pey =  diff_predictor(Pe ,"y", corde,f ) #  Diferencial del campo "B" Magnetico en la coordenada  y
    gradiente_presion_electron = diff_Pex +diff_Pey
    o_ = 0.1*(e**2*n_e_c)/(me*Ce*(n_e_c*Qei + n_n_c*Qen3100))
    densidad_corriente = E*o_ + tenedor*gradiente_presion_electron
    for j in range(1,f+1):
                          for i in range(1,f+1):
                                      intensidad_corriente[j][i] = densidad_corriente[j][i]*delta_x[j][i]*delta_y[j][i] 
                                      B[j][i]=  u_o*intensidad_corriente[j][i]/2*pi*delta_y[j][i]
    
    #-----------------------------------------------------------------------------------------------------
    
    diff_xx_x =  diff_corrector(u_temp ,"x", corde,f )
    diff_xx_y =  diff_corrector(v_temp ,"y", corde,f )
    diff_xy_x =  diff_corrector(v_temp , "x", corde,f )
    diff_xy_y =  diff_corrector(u_temp , "y", corde,f )
    diff_yy_x =  diff_corrector(u_temp , "x", corde,f )
    diff_yy_y =  diff_corrector(v_temp , "y", corde,f )
    diff_gx =  diff_corrector(Tg_temp , "x", corde,f ) #  "Tg" Temperatura de gas, "corde" toda las coordenadas de la malla 
    diff_gy =  diff_corrector(Tg_temp , "y", corde,f )
    diff_Bx =  diff_corrector(B ,"x", corde,f ) #  Diferencial del campo "B"  Magnetico en la coordenada x
    diff_By =  diff_corrector(B ,"y", corde,f ) #  Diferencial del campo "B" Magnetico en la coordenada  y
    
    
    vis = (0.5*ma*Cg)/2**(1/2)*( n_n_c/(n_n_c*Qnn + n_e_c*Qin)+ n_e_c/(n_n_c*Qin + n_e_c*Qii)) #  calculo de la viscosidad
    Kg = Sk*(k**2*Tg_temp/ma*Cg)*( n_n_c/(n_n_c*Qnn + n_e_c*Qin)+ n_e_c/(n_e_c*Qii +n_n_c*Qin)) # "Kg" factor de transmision del calor del gas , "Cg"  velocidad termica del gas
    q_gx = -Kg*(diff_gx) #   "q_gx"  Difusion del calor del gas en la coordenada x
    q_gy = -Kg*(diff_gy) #   "q_gy"  Difusion del calor del gas en la coordenada y
    El = ( 3*U2/ma)*(n_e_c*Ce*Qei + n_e_c*Ce*Qen )*k*(Te - Tg_temp)
    estres_xx = (4/3)*vis*(diff_xx_x) - (2/3)*vis*(diff_yy_y) 
    estres_xy = vis*(diff_xy_y) + vis*(diff_xy_x) 
    estres_yy = (4/3)*vis*(diff_yy_y) - (2/3)*vis*(diff_yy_x) 
    
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    
    
    #----------------------------------------------  7  Redefinir  F G  S en   base al U_temp -----------------------------------------------------------------------------------------------------#
    F1= U3_temp
    F3= (U3_temp*U3_temp)/U1_temp +presion + (B*B)/2*u_o  - estres_xx  # "u_o" Permeabilidad en el vacio H/m
    F4= (U3_temp*U4_temp)/U1_temp - estres_xy
    F5= U3_temp*Hg -U3_temp/U1_temp*estres_xx - U4_temp/U1_temp*estres_xy + q_gx # "q_gx"  difusion del calor del gas en la coordenada x
    
    G1= U4_temp
    G3= (U3_temp*U4_temp)/U1_temp - estres_xy
    G4= (U4_temp*U4_temp)/U1_temp + presion + (B**2)/2*u_o - estres_yy 
    G5= U4_temp*Hg - U3_temp/U1_temp*estres_xy - U4_temp/U1_temp*estres_yy + q_gy #  "q_gy" difusion del calor del gas en la coordenada y
    
    S1= 0
    S3= 0
    S4= 0                                                
    S5= El                                               #   "El" energia de transferencia colisionale entre los electrones y las particulas pesadas
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    
    
    ######################  4.- INICIO  DEL  CORRECTOR  ESPECIES PESADAS #################################
   
    #-----------------------------------------  1   Derivativo de  F   G   S    del corrector   --------------------------------------------------------------------------------------------------#
    F1_corrector = diff_corrector(F1 ,"x", corde,f ) 
    F3_corrector = diff_corrector(F3 ,"x", corde,f )
    F4_corrector = diff_corrector(F4 ,"x", corde,f )
    F5_corrector = diff_corrector(F5 ,"x", corde,f )
    
    G1_corrector = diff_corrector(G1 ,"y", corde,f )
    G3_corrector = diff_corrector(G3 ,"y", corde,f )
    G4_corrector = diff_corrector(G4 ,"y", corde,f )
    G5_corrector = diff_corrector(G5 ,"y", corde,f )
    
    diff_np1_U1 = S1 - F1_corrector -  G1_corrector
    diff_np1_U3 = S3 - F3_corrector -  G3_corrector
    diff_np1_U4 = S4 - F4_corrector -  G4_corrector
    diff_np1_U5 = S5 - F5_corrector -  G5_corrector
    
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    
    #------------------------------------------  2   Suavizador  del  corrector  ----------------------------------------------------------------------------------------------------------------# 
    for j in range(0,f-1):
      for i in range(0,f-1):
        D1[j][i]= -unod4x*(  U1_temp[j][i+2] + U1_temp[j][i-2] - 4*( U1_temp[j][i+1]+U1_temp[j][i-1] )+6*U1_temp[j][i] -  unod4y*( U1_temp[j+2][i]+U1_temp[j-2][i]- 4*( U1_temp[j+1][i]+U1_temp[j-1][i] )+ 6*U1_temp[j][i]) + unod2x*( U1_temp[j][i+1] -2*U1_temp[j][i]+U1_temp[j][i-1]) + unod2y*( U1_temp[j+1][i] - 2*U1_temp[j][i]+ U1_temp[j-1][i] ))
        D3[j][i]= -tresd4x*( U3_temp[j][i+2] + U3_temp[j][i-2] - 4*( U3_temp[j][i+1]+U3_temp[j][i-1] )+6*U3_temp[j][i] - tresd4y*( U3_temp[j+2][i]+U3_temp[j-2][i]- 4*( U3_temp[j+1][i]+U3_temp[j-1][i] )+ 6*U3_temp[j][i]) + tresd2x*( U3_temp[j][i+1] -2*U3_temp[j][i]+U3_temp[j][i-1]) + tresd2y*( U3_temp[j+1][i] - 2*U3_temp[j][i]+ U3_temp[j-1][i] ))
        D4[j][i]= -cuatrod4x*( U4_temp[j][i+2] + U4_temp[j][i-2] - 4*( U4_temp[j][i+1]+U4_temp[j][i-1] )+6*U4_temp[j][i] - cuatrod4y*( U4_temp[j+2][i]+U4_temp[j-2][i]- 4*( U4_temp[j+1][i]+U4_temp[j-1][i] )+ 6*U4_temp[j][i]) + cuatrod2x*( U4_temp[j][i+1] -2*U4_temp[j][i]+U4_temp[j][i-1]) + cuatrod2y*( U4_temp[j+1][i] - 2*U4_temp[j][i]+ U4_temp[j-1][i] ))
        D5[j][i]= -cincod4x*( U5_temp[j][i+2] + U5_temp[j][i-2] - 4*( U5_temp[j][i+1]+U5_temp[j][i-1])+6*U5_temp[j][i] - cincod4y*( U5_temp[j+2][i]+U5_temp[j-2][i]- 4*( U5_temp[j+1][i]+U5_temp[j-1][i] )+ 6*U5_temp[j][i]) + cincod2x*( U5_temp[j][i+1] -2*U5_temp[j][i]+U5_temp[j][i-1]) + cincod2y*( U5_temp[j+1][i] - 2*U5_temp[j][i]+ U5_temp[j-1][i]  ))
        
   #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------# 
   #----------------------------------------------  3   calculo  final de  U --------------------------------------------------------------------------------------------------------------------#
   
    for j in range(1,f+1):
      for i in range(2,f+1):    
        U1[j][i]=U1[j][i]+0.5*dt*(diff_n_U1[j][i]+diff_np1_U1[j][i])+D1[j][i]
        U3[j][i]=U3[j][i]+0.5*dt*(diff_n_U3[j][i]+diff_np1_U3[j][i])+D3[j][i]
        U4[j][i]=U4[j][i]+0.5*dt*(diff_n_U4[j][i]+diff_np1_U4[j][i])+D4[j][i]
        U5[j][i]=U5[j][i]+0.5*dt*(diff_n_U5[j][i]+diff_np1_U5[j][i])+D5[j][i]
        
   #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
   
   #----------------------------------------------   4   Condiciones frontera --------------------------------------------------------------------------------------------------------------------#
   
   #----------------------------------- LINEA  INICIAL ------------------------------
    for t in range(f):
            if 1 <= i <= 25:
                U1[0][t]= 0.1
                U3[0][t]= 0.1
                U4[0][t]= 0.1
                U5[0][t]= 0.1
                
            elif 25 < i <= 29:
                U1[0][t]= 0.1
                U3[0][t]= 0.1
                U4[0][t]= 0.1
                U5[0][t]= 0.1
               
            elif 29 < i <= 60:
                U1[0][t]= 0.1
                U3[0][t]= 0.1
                U4[0][t]= 0.1
                U5[0][t]= 0.1
                
            else:
                U1[0][t]= 0.1
                U3[0][t]= 0.1
                U4[0][t]= 0.1
                U5[0][t]= 0.1
        
        #-------------------------------------------------------------------#
        #----------------LINEA   FINAL -------------------------------------# 
    for w in range(f):
              if  0 <= j <= 15:
                  U1[f][w]= 0.1
                  U3[f][w]= 0.1
                  U4[f][w]= 0.1
                  U5[f][w]= 0.1
                    
              elif 15 < j <= 32:
                  U1[f][w]= 0.1
                  U3[f][w]= 0.1
                  U4[f][w]= 0.1
                  U5[f][w]= 0.1
                   
              elif 32 < j <= 60:
                  U1[f][w]= 0.1
                  U3[f][w]= 0.1
                  U4[f][w]= 0.1
                  U5[f][w]= 0.1
              else:
                  U1[f][w]= 0.1
                  U3[f][w]= 0.1
                  U4[f][w]= 0.1
                  U5[f][w]= 0.1
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        
    #-------------------------------------   5  Calculo de las variables libres en  base a U  final anterior  -------------------------------------------------------------------#
    #----------------------------------------  Calculando  Tg  y Te ------------------------------------------------------------------------------------------------------------#
    u = U3/U1
    v = U4/U1
    Energia_gas = U5/U1
    Energia_electron = U6/U2
    Tg = 0.0032031*(U5/U1 - 0.5*(U3*U3/U1*U1 - U4*U4/U1*U1))
     
   
   
   
    #################################### BUCLE  DE LOS ELECTRONES   ###############################################################################
   
    ###########################################  INICIO DEL PREDICTOR ELECTRONES  #############################################################
   
    cte = datetime.datetime.now()
    
    #while dt > dt_electrones  :
    
    
    vuelta_electrones = 0
    for time_electrones in range(1):
       print("Esta es la vuelta ELECTRONES ", vuelta_electrones)
       
       #------------------------------------------- 1   Definir  los terminos F  G  S  en base   a   U   -------------------------------------------------------------------------------------------------#
       
       
       Ce =  (abs(k*Te/me))**(1/2)                                      #  Velocidad termica de los electrones
       n_n_p = abs(U1*Ra)/k                                                #  densidad elemental de las particulas neutrales x volumen
       n_e_p = abs(U2*Re)/k                                                #  densidad elemental de electrones  x volumen
       V  = 1.239e7*(abs(Te/n_e_p))**(1/2)                              #   Facctor sparta para el calculo de las seccion de colision
       for j in range(1,f+1):
                         for i in range(1,f+1):
                                 Qen3100[j][i] = (-0.488+3.96*10**(-4)*Te[j][i])*10**(-20)                  # "Qen3100"  si  la temperatura es mayor a 3000  grados
                                 Qei[j][i] =  ((e**4)*log(V[j][i]))/24*pi*(e_o*k*Te[j][i])**2                #  "e_o" permitividad en el vacio , "Te" Temperatura del electron, "V" factor sparta, "k" Constante de Botzman
                                 Qin = 1.4*10**(-18)   #  Seccion de colision entre ion y neutron en metros cuadrados
                                 Qen[j][i] = (0.713 - 4.5*10**(-4)*Te[j][i] + 1.5*10**(-7)*Te[j][i]**2)*10**(-20) #  Seccion de colision cruzada de electron y neutron
                                 Qee[j][i] = abs(Qei[j][i])
                                 Qee[j][i] =  Qei[j][i]
       
       Qen3100 = abs(Qen3100)
       Qei = abs(Qei)
       tenedor = e/(me*Ce*(n_e_p*Qei + n_n_p*Qen3100))  #  Para el calculo de la corriente
       Pe = U2*Te*Re                                    #  presion del electron
       diff_Pex =  diff_predictor(Pe ,"x", corde,f ) #  Diferencial de la presion  Pe
       diff_Pey =  diff_predictor(Pe ,"y", corde,f ) #  Diferencial del la presion Pe en la coordenada   y
       gradiente_presion_electron = diff_Pex +diff_Pey  # Gradiente de presion
       o_ = 0.1*(e**2*n_e_p)/(me*Ce*(n_e_p*Qei + n_n_p*Qen3100))#  Conductividad electrica
       densidad_corriente = E*o_ + tenedor*gradiente_presion_electron
       for j in range(1,f+1):                           # Calculo del campo magnetico
                         for i in range(1,f+1):
                                  intensidad_corriente[j][i] = densidad_corriente[j][i]*delta_x[j][i]*delta_y[j][i] 
                                  B[j][i]=  u_o*intensidad_corriente[j][i]/2*pi*delta_y[j][i]
       
       diff_ex =  diff_predictor(Te , "x", corde,f ) # Diferencial de la temperatura Te
       diff_ey =  diff_predictor(Te , "y", corde,f ) #  Diferencial de la Te
       diff_Bx =  diff_predictor(B ,"x", corde,f ) #  Diferencial del campo "B"  Magnetico en la coordenada x
       diff_By =  diff_predictor(B ,"y", corde,f ) #  Diferencial del campo "B" Magnetico en la coordenada  y
       
       Ce1= 3**(1/2)*Ce            # Velocidad termica del electron especial
       R = (1.09*10**(-20))/abs(Te)**(9/2)          # Para calcular n_e     
       S =  (21.60*10**14)*( abs(Te)**(3/2) )*(2.718**(-1*( 183200.57/Te ) )) # "Ei" Energia de ionizacion y Te Temperatura del electron, "h" Constante de Planck, "k" Constante Boltzman
       
       
       Ke = Sk*(n_e_p*k**2*Te)/me*Ce*(n_e_p*Qei + n_n_p*Qen3100)
       q_ex = -Ke*(diff_ex) #   "q_ex"   Difusion del calor de los electronjmes en la coordenada x
       q_ey = -Ke*(diff_ey) #   "q_ey"   Difusion del calor de la nube de electrones en la coordenada y    
       ne_hh_p = R*n_e_p*( S*n_n_p - (n_e_p)**2 )       #  "ne_hh" es la nube de electrones  producto de la colision
       
       jx = (1/u_o)*(diff_By)     #  "jx" densidad de corriente en el eje x
       jy = (-1/u_o)*(diff_Bx)     #  "jy" densidad de corriente en el eje y
       
       v_e = U4/U1 - jy/e*n_e_p
       u_e = U3/U1 - jx/e*n_e_p
       
       v_e = abs(v_e)
       u_e = abs(u_e)
       
       vis = (0.5*me*Ce)/(2**(1/2))*(n_n_p/(n_n_p*Qen + n_e_p*Qei) + n_e_p/(n_n_p*Qei + n_e_p*Qee) ) #  calculo de la viscosidad
       
       
       El = ( (3*U2)/ma)*(n_e_p*Ce*Qei +  n_e_p*Ce*Qen)*k*(Te -Tg)
       Da = (abs(pi*k*Tg/4*ma)**(1/2))*(1+ Te/Tg)/Qin*(n_e_p + n_n_p)       #  Ambipolar difusion , "n_e" densidad de electrones, "n_n" densidad de neutrone
       
       diff_U2y =  diff_predictor(U2,"y", corde,f )
       K = Da*diff_U2y                   #  factor de Ambipolar difusion, que  forma parte de "S2"
       K = diff_predictor(K,"y", corde,f ) # Segunda derivada del termino K
       
       #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
       
       F2 = (U2*U3)/U1
       G2= (U2*U4)/U1 
       S2= abs(me*ne_hh_p) + K      #  "K" factor de Ambipolar difusion  ,  "ne_hh" es la nube de electrones  producto de la colision 
       
       
       F6= -jx*(Ei/e + (ceA*k*Te)/(ceA -1)*e ) + q_ex      #  "Ei"  energia de ionizacion ,"ceA" calor especifico del Argon , "q_ex" difusion del calor de la nube de electrones en la coordenada x
       G6= -jy*(Ei/e + (ceA*k*Te)/(ceA -1)*e ) +q_ey        #  "q_ey"  difusion del calor de la nube de electrones en la coordenada y
       S6= (densidad_corriente**2)/o_ - El     
       
       
      #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------# 
      
      #---------------------------------------------    2     Derivativa predictiva de  F   G   S      ---------------------------------------------------------------------------------------------#
       
       F2_predictor = diff_predictor(F2 ,"x", corde,f )
       G2_predictor = diff_predictor(G2 ,"y", corde,f )
       F6_predictor = diff_predictor(F6 ,"x", corde,f )
       G6_predictor = diff_predictor(G6 ,"y", corde,f )
       
       diff_n_U2 = S2 - F2_predictor -  G2_predictor
       diff_n_U6 = S6 - F6_predictor -  G6_predictor
       
      #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
      
      #----------------------------------------------   3   Calcular el tiempo y suavizador  -------------------------------------------------------------------------------------------------------#
       d_electron_all = np.array([[0.0001  for col in range(f)] for row in range(f)])
       delta_x = delta( "x", corde,f )
       delta_y = delta( "y", corde,f )
       delta_y = abs(delta_y)
       for i in range(1,f+1):    
         for w in range(1,f+1):   
            c_factor = (ceA*Re*Te[i][w])**(1/2)
            Re_delta_x = (U2[i][w]*u_e[i][w]*delta_x[i][w])/vis[i][w]
            Re_delta_y = (U2[i][w]*v_e[i][w]*delta_y[i][w])/vis[i][w]
            Re_delta = min( Re_delta_x, Re_delta_y)
            delta_t_CFL = (u_e[i][w]/delta_x[i][w] + v_e[i][w]/delta_y[i][w] +  c_factor*(1/(delta_x[i][w])**2+1/delta_y[i][w]**2)**(1/2) )**(-1)
            delta_tiempo = (factor_seguridad_tiempo*delta_t_CFL)/(1+ 2/Re_delta)
            d_electron_all[i-1][w-1] = delta_tiempo 
            dt_electrones = np.amin(d_electron_all)
       
       print(" Tiempo de los electron ",dt_electrones)
      
       
       #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
       
       #----------------------------------------------- 4   Definir  U_temp   -----------------------------------------------------------------------------------------------------------------------#
       U2_temp=U2+dt_electrones*diff_n_U2
       U6_temp=U6+dt_electrones*diff_n_U6
      
       
       
       #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
              
       #-----------------------------------------------  5  Condiciones frontera  -------------------------------------------------------------------------------------------------------------------#
       #-------------------------------LINEA  0 ----------------------
       for q in range(f):
           if 1 <= i <= 25:
               U2_temp[0][q]= 0.0001
               U6_temp[0][q]= 0.0001
               
           elif 25 < i <= 29:
               U2_temp[0][q]= 0.0001
               U6_temp[0][q]= 0.0001
               
           elif 29 < i <= 60:
               U2_temp[0][q]= 0.0001
               U6_temp[0][q]= 0.0001
               
           else:
               U2_temp[0][q]= 0.0001
               U6_temp[0][q]= 0.0001
       
       #-------------------------------------------------------------------#
       #----------------LINEA   FINAL -------------------------------------# 
       for r in range(f):
               if  0 <= i <= 15:
                   U2_temp[f][r]= 0.0001
                   U6_temp[f][r]= 0.0001
                   
               elif 15 < i <= 32:
                   U2_temp[f][r]= 0.0001
                   U6_temp[f][r]= 0.0001
                  
               elif 32 < i <= 60:
                   U2_temp[f][r]= 0.0001
                   U6_temp[f][r]= 0.0001
                   
               else:
                   U2_temp[f][r]= 0.0001
                   U6_temp[f][r]= 0.0001
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
   
       #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#       
       
       #------------------------------------------------------------ 6   Definir  la variables libres en bases a U_temp -----------------------------------------------------------------------------#
       sello = e/k*e_o
       gradiente_campo_E = sello*(U1*Ra - U2_temp*Re )
       densidad_carga = gradiente_campo_E *e_o
       carga = densidad_carga*delta_x*delta_y
       for j in range(1,f+1):
                         for i in range(1,f+1):
                                      u_e[j][i] =  ( abs(2*E[j][i-1]*carga[j][i-1]/me)  + u_e[j][i-1]*u_e[j][i-1])**(1/2)
    
       
       Te_temp = ( U6_temp/U2_temp - (u_e*u_e)/2 )/1.5*Re
       Te_temp = abs(Te_temp)
       
       
       #----------------------------------------  calculo de  B temporal -----------------------------------------------------------------#                                        
       Ce =  (abs(k*Te_temp/me))**(1/2)
       n_n_p = (U1*Ra)/k                                                #  densidad de las particulas neutrales
       n_e_p = (U2_temp*Re)/k   
       V  = 1.239e7*(abs(Te_temp/n_e_p))**(1/2)
       for j in range(1,f+1):
                          for i in range(1,f+1):

                                        Qen3100[j][i] = (-0.488+3.96*10**(-4)*Te_temp[j][i])*10**(-20)                  # "Qen3100"  si  la temperatura es mayor a 3000  grados
                                        Qei[j][i] =  ((e**4)*log(V[j][i]))/24*pi*(e_o*k*Te_temp[j][i])**2                #  "e_o" permitividad en el vacio , "Te" Temperatura del electron, "V" factor sparta, "k" Constante de Botzman
                                        Qin = 1.4*10**(-18)   #  Seccion de colision entre ion y neutron en metros cuadrados
                                        Qen[j][i] = (0.713 - 4.5*10**(-4)*Te_temp[j][i] + 1.5*10**(-7)*Te_temp[j][i]**2)*10**(-20) #  Seccion de colision cruzada de electron y neutron
                                        Qee[j][i] =  Qei[j][i]
       tenedor = e/(me*Ce*(n_e_p*Qei + n_n_p*Qen3100))
       Pe = U2_temp*Te_temp*Re
       diff_Pex =  diff_predictor(Pe ,"x", corde,f ) #  Diferencial del campo "B" Magnetico en la coordenada  y
       diff_Pey =  diff_predictor(Pe ,"y", corde,f ) #  Diferencial del campo "B" Magnetico en la coordenada  y
       gradiente_presion_electron = diff_Pex +diff_Pey
       o_ = 0.1*(e**2*n_e_p)/(me*Ce*(n_e_p*Qei + n_n_p*Qen3100))
       densidad_corriente = E*o_ + tenedor*gradiente_presion_electron
       
       for j in range(1,f+1):
                           for i in range(1,f+1):
                                    intensidad_corriente[j][i] = densidad_corriente[j][i]*delta_x[j][i]*delta_y[j][i] 
                                    B[j][i]=  u_o*intensidad_corriente[j][i]/2*pi*delta_y[j][i]
       #-------------------------------------------------------------------------------------------------------------------------------------#                                        
       diff_ex =  diff_predictor(Te_temp , "x", corde,f )
       diff_ey =  diff_predictor(Te_temp , "y", corde,f )
       diff_Bx =  diff_predictor(B ,"x", corde,f ) #  Diferencial del campo "B"  Magnetico en la coordenada x
       diff_By =  diff_predictor(B ,"y", corde,f ) #  Diferencial del campo "B" Magnetico en la coordenada  y
       jx = (1/u_o)*(diff_By)     #  "jx" densidad de corriente en el eje x
       jy = (-1/u_o)*(diff_Bx)     #  "jy" densidad de corriente en el eje y
     
       R = (1.09e-20)/(abs(Te_temp))**(9/2)          # Para calcular n_e     
       S =  (21.60*10**14)*( (Te_temp)**(3/2) )*(2.718**(-1*( 183200.57/Te_temp ) ))         # "Ei" Energia de ionizacion y Te Temperatura del electron, "h" Constante de Planck, "k" Constante Boltzman
       ne_hh_p = R*n_e_p*( S*n_n_p - (n_e_p)**2 )       #  "ne_hh" es la nube de electrones  producto de la colision
       vis = (0.5*me*Ce)/(2**(1/2))*(n_n_p/(n_n_p*Qen + n_e_p*Qei) + n_e_p/(n_n_p*Qei + n_e_p*Qee) ) #  calculo de la viscosidad
       El = ( (3*U2_temp)/ma)*(n_e_p*Ce*Qei +  n_e_p*Ce*Qen)*k*(Te_temp -Tg)
       Da = (abs(pi*k*Tg/4*ma)**(1/2))*(1+ Te_temp/Tg)/Qin*(n_e_p + n_n_p)       #  Ambipolar difusion , "n_e" densidad de electrones, "n_n" densidad de neutrone
       diff_U2y =  diff_predictor(U2_temp,"y", corde,f )
       K = Da*diff_U2y                   #  factor de Ambipolar difusion, que  forma parte de "S2"
       K = diff_predictor(K,"y", corde,f ) # Segunda derivada del termino K
       n_n_p= abs(n_n_p)
       #  Revisar  los valores de  Qei  y Qen , porque  tiene  valores imposibles de tamaño  de superficie
       Ke = Sk*(n_e_p*k**2*Te_temp)/me*Ce*(n_e_p*Qei + n_n_p*Qen)
       print("Ce",Ce)
       print("n_e_p",n_e_p)
       print("n_n_p",n_n_p)
       print("Te_temp",Te_temp)
       print("Qei",Qei)
       print("Qen",Qen)
       print(" Ke",Ke)
       
       q_ex = -Ke*(diff_ex) #   "q_ex"   Difusion del calor de los electrones en la coordenada x
       q_ey = -Ke*(diff_ey) #   "q_ey"   Difusion del calor de la nube de electrones en la coordenada y    
       
       #---------------------------------------------------------- 7  Redefinir   F G S  en base  U_temp -------------------------------------------------------------------#
       F2 = (U2_temp*U3)/U1
       G2= (U2_temp*U4)/U1 
       S2= me*ne_hh_p + K                                     #  "K" factor de Ambipolar difusion  ,  "ne_hh" es la nube de electrones  producto de la colision 
       
       F6= -jx*(Ei/e + (ceA*k*Te_temp)/(ceA -1)*e ) + q_ex      #  "Ei"  energia de ionizacion ,"ceA" calor especifico del Argon , "q_ex" difusion del calor de la nube de electrones en la coordenada x
       G6= -jy*(Ei/e + (ceA*k*Te_temp)/(ceA -1)*e ) +q_ey        #  "q_ey"  difusion del calor de la nube de electrones en la coordenada y
       S6= (densidad_corriente**2)/o_ - El    
       #--------------------------------------------------------------------------------------------------------------------------------------------------------------------#
       
       ###############################################   INICIO   CORRECTOR   ELECTRONES   ################################################################################
       
       
       #-------------------------------------------------- 1  Derivativo correctivo de los  terminos  F   G S J --------------------------------------------------------------------#
       F2_corrector = diff_corrector(F2 ,"x", corde,f )
       F6_corrector = diff_corrector(F6 ,"x", corde,f )
       
       G2_corrector = diff_corrector(G2 ,"y", corde,f )
       G6_corrector = diff_corrector(G6 ,"y", corde,f ) 
       
       diff_n1_U2 = S2 - F2_corrector -  G2_corrector
       diff_n1_U6 = S6 - F6_corrector -  G6_corrector
       
       
       #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
       
       #----------------------------------------------------2  suavizador  corrector para electrones -----------------------------------------------------------------------------------#
       for j in range(0,f-1):
         for i in range(0,f-1):
           D2[j][i]= -dosd4x*( U2_temp[j][i+2] + U2_temp[j][i-2] - 4*( U2_temp[j][i+1]+U2_temp[j][i-1] )+6*U2_temp[j][i] - dosd4y*( U2_temp[j+2][i]+U2_temp[j-2][i]- 4*( U2_temp[j+1][i]+U1_temp[j-1][i] )+ 6*U2_temp[j][i]) + dosd2x*( U2_temp[j][i+1] -2*U2_temp[j][i]+U2_temp[j][i-1]) + dosd2y*( U2_temp[j+1][i] - 2*U2_temp[j][i]+ U2_temp[j-1][i] ))
           D6[j][i]= -seisd4x*( U6_temp[j][i+2] + U6_temp[j][i-2] - 4*( U6_temp[j][i+1]+U6_temp[j][i-1])+6*U6_temp[j][i]- seisd4y*( U6_temp[j+2][i]+U6_temp[j-2][i]- 4*( U6_temp[j+1][i]+U6_temp[j-1][i] )+ 6*U6_temp[j][i]) + seisd2x*( U6_temp[j][i+1] -2*U6_temp[j][i]+U6_temp[j][i-1]) + seisd2y*( U6_temp[j+1][i] - 2*U6_temp[j][i]+ U6_temp[j-1][i] ))
      
      #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
      
      #----------------------------------------------------- 3  Calculo  final    (CORRECTOR+PREDICTOR)*DT ELECTRONES -------------------------------------------------------------------# 
       
       for j in range(1,f+1):
         for i in range(2,f+1):    
           U2[j][i]=U2[j][i]+0.5*dt_electrones*(diff_n1_U2[j][i]+diff_n_U2[j][i])+D2[j][i]
           U6[j][i]=U6[j][i]+0.5*dt_electrones*(diff_n1_U6[j][i]+diff_n_U6[j][i])+D6[j][i]
       
       
       
      #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
      
      #----------------------------------------------------- 4   Condiciones de Frontera --------------------------------------------------------------------------------------------------#        
      #-------------------LINEA  INICIAL ------------------------------# 
       for w in range(f):
               if 1 <= i <= 25:
                   U2[0][t]= 0.1
                   U6[0][t]= 0.1
                   
               elif 25 < i <= 29:
                   U2[0][t]= 0.1
                   U6[0][t]= 0.1
                  
               elif 29 < i <= 60:
                   U2[0][t]= 0.1
                   U6[0][t]= 0.1
                   
               else:
                   U2[0][t]= 0.1
                   U6[0][t]= 0.1
           
       #---------------------------------------------------------------#
       
       #----------------LINEA   FINAL ---------------------------------# 
       for w in range(f):
                 if  0 <= j <= 15:
                     U2[f][w]= 0.1
                     U6[f][w]= 0.1
                       
                 elif 15 < j <= 32:
                     U2[f][w]= 0.1
                     U6[f][w]= 0.1
                      
                 elif 32 < j <= 60:
                     U2[f][w]= 0.1
                     U6[f][w]= 0.1
                 else:
                     U2[f][w]= 0.1
                     U6[f][w]= 0.1
           
    #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    
    
    #----------------------------------------------------------  5   Calculo de  las variables libres en base  U -------------------------------------------------------------------------#
    
       Energia_electron = U6/U2
       #  Calcular de nuevo el campo electrico  para variar la velocidad
       
       sello = e/k*e_o
       gradiente_campo_E = sello*(U1*Ra - U2*Re )
       densidad_carga = gradiente_campo_E *e_o
       carga = densidad_carga*delta_x*delta_y
       
       for j in range(1,f+1):
                  for i in range(1,f+1):
                                      u_e[j][i] =  ( abs(2*E[j][i-1]*carga[j][i-1]/me)  + u_e[j][i-1]*u_e[j][i-1])**(1/2)
       Te = ( U6/U2 - (u_e*u_e)/2 )/1.5*Re
       Te =  abs(Te)
       vuelta_electrones = vuelta_electrones+1
       
       
    #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    
    
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
    cte1 = datetime.datetime.now()
   #----------------------  CALCULO DEL ERROR  DE PRESION ---------------------------------------------------------------------------------------------------#
    '''
    acumular = 0
    asta = 0
    for i in range(0,f):
      for j in range(0,f):
                     acumular= acumular + (presion[i][j] - presion[i][j+1])**2
      bandera= acumular/f
      asta = asta+ bandera 
      acumular = 0
      
    error_horizontal = asta/f 
    error = error_horizontal
    '''
    
    
    
    #----------------------------------------  GUARDO LOS RESULTADOS  FINALES ------------------------------------------------------------------------------#
           
    U1_completo.extend(U1)
    U2_completo.extend(U2)
    U3_completo.extend(U3)
    #S1_completo.extend(S1)
    S2_completo.extend(S2)
    #S3_completo.extend(S3)
    F1_completo.extend(F1)
    F2_completo.extend(F2)
    F3_completo.extend(F3)
       
    U1_completo.extend(U1)
    U2_completo.extend(U2)
    U3_completo.extend(U3)
    U4_completo.extend(U4)
    U5_completo.extend(U5)
    U6_completo.extend(U6)
        
    F1_completo.extend(F1)
    F2_completo.extend(F2)
    F3_completo.extend(F3)
    F4_completo.extend(F4)
    F5_completo.extend(F5)
    F6_completo.extend(F6)
        
    G1_completo.extend(G1)
    G2_completo.extend(G2)
    G3_completo.extend(G3)
    G4_completo.extend(G4)
    G5_completo.extend(G5)
    G6_completo.extend(G6)
        
    #S1_completo.extend(S1)
    S2_completo.extend(S2)
    #S3_completo.extend(S3)
    #S4_completo.extend(S4)
    #S5_completo.extend(S5)
    #S6_completo.extend(S6)

cts = datetime.datetime.now()    
(f,f,4000)

cts1 = datetime.datetime.now()    

print("##############################################")
ct1 = datetime.datetime.now()    
cte1 = datetime.datetime.now()    
bucle = open("DATA/BUCLES.txt","a")
bucle.write(" BUCLE  GASES  "+str(time))
bucle.write("  "+str(ct))
bucle.write("----")
bucle.write( str(ct1)+'   ')
bucle.write(" BUCLE  ELECTRONES  "+str(vuelta_electrones))
bucle.write(" "+str(cte))
bucle.write("====")
bucle.write( str(ct1)+'   ')
bucle.write(" SOR CALCULO  "+str(vuelta_electrones))
bucle.write(" "+str(cts))
bucle.write(str(cts1)+'\n')
bucle.close()


##############################################################   FIN  DEL BUCLE PRINCIPAL ##########################################



################################# 5.-  INICIO  VISUALIZACION ALMACENAMIENTO  DE  RESULTADOS  #######################################


resultado1 = open("DATA/U1.txt","w")
resultado1.write(str(U1_completo))
resultado1.close()
resultado2 = open("DATA/U2.txt","w")
resultado2.write(str(U2_completo))
resultado2.close()
resultado3 = open("DATA/U3.txt","w")
resultado3.write(str(U3_completo))
resultado3.close()
#-----------------------------------------------------------------
resultado4 = open("DATA/F1.txt","w")
resultado4.write(str(F1_completo))
resultado4.close()
resultado5 = open("DATA/F2.txt","w")
resultado5.write(str(F2_completo))
resultado5.close()
resultado6 = open("DATA/F3.txt","w")
resultado6.write(str(F3_completo))
resultado6.close()
#-----------------------------------------------------------------
resultado7 = open("DATA/S1.txt","w")
resultado7.write(str(S1_completo))
resultado7.close()
resultado8 = open("DATA/S2.txt","w")
resultado8.write(str(S2_completo))
resultado8.close()
resultado9 = open("DATA/S3.txt","w")
resultado9.write(str(S3_completo))
resultado9.close()

resultado1 = open("DATA/U1.txt","w")
resultado1.write(str(U1_completo))
resultado1.close()
resultado2 = open("DATA/U2.txt","w")
resultado2.write(str(U2_completo))
resultado2.close()
resultado3 = open("DATA/U3.txt","w")
resultado3.write(str(U3_completo))
resultado3.close()
resultado4 = open("DATA/U4.txt","w")
resultado4.write(str(U4_completo))
resultado4.close()
resultado5 = open("DATA/U5.txt","w")
resultado5.write(str(U5_completo))
resultado5.close()
resultado6 = open("DATA/U6.txt","w")
resultado6.write(str(U6_completo))
resultado6.close()

resultado7 = open("DATA/F1.txt","w")
resultado7.write(str(F1_completo))
resultado7.close()
resultado8 = open("DATA/F2.txt","w")
resultado8.write(str(F2_completo))
resultado8.close()
resultado9 = open("DATA/F3.txt","w")
resultado9.write(str(F3_completo))
resultado9.close()
resultado10 = open("DATA/F4.txt","w")
resultado10.write(str(F4_completo))
resultado10.close()
resultado11 = open("DATA/F5.txt","w")
resultado11.write(str(F5_completo))
resultado11.close()
resultado12 = open("DATA/F6.txt","w")
resultado12.write(str(F6_completo))
resultado12.close()

resultado13 = open("DATA/G1.txt","w")
resultado13.write(str(G1_completo))
resultado13.close()
resultado14 = open("DATA/G2.txt","w")
resultado14.write(str(G2_completo))
resultado14.close()
resultado15 = open("DATA/G3.txt","w")
resultado15.write(str(G3_completo))
resultado15.close()
resultado16 = open("DATA/G4.txt","w")
resultado16.write(str(G4_completo))
resultado16.close()
resultado17 = open("DATA/G5.txt","w")
resultado17.write(str(G5_completo))
resultado17.close()
resultado18 = open("DATA/G6.txt","w")
resultado18.write(str(G6_completo))
resultado18.close()

resultado19 = open("DATA/S1.txt","w")
resultado19.write(str(S1_completo))
resultado19.close()
resultado20 = open("DATA/S2.txt","w")
resultado20.write(str(S2_completo))
resultado20.close()
resultado21 = open("DATA/S3.txt","w")
resultado21.write(str(S3_completo))
resultado21.close()
resultado22 = open("DATA/S4.txt","w")
resultado22.write(str(S4_completo))
resultado22.close()
resultado23 = open("DATA/S5.txt","w")
resultado23.write(str(S5_completo))
resultado23.close()
resultado24 = open("DATA/S6.txt","w")
resultado24.write(str(S6_completo))
resultado24.close()

