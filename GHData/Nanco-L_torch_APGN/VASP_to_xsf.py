# Code for generating xsf file using VASP file 
# xsf file contains structural information(cell, coordination, atom type)
#  and energy, force.
# With this process, we compress the data in CONTCAR, XDATCAR and OUTCAR
# In addition, file reading process in fortran code can be easier.
# ==========================================================================
# The code contains...
#  props class: data structure which contains all informations for xsf file
#  property pickup function (OneForcefromOUTCAR and OneEnergyfromOUTCAR)
# --------------------------------------------------------------------------
# Usage
#  python VASP_to_xsf.py [input_file]
#  - input_file contains the information of directory
#    CONTCAR (or XDATCAR) and OUTCAR should be in the directory
#  - each line in input files are...
#    [directory] [structure_file_type] [total_step] [step_interval]
#    * directory: directory contains VASP files
#    * structure_file_type: CONTCAR or XDATCAR
#    * total_step: for XDATCAR, total MD steps
#    * step_interval: for XDATCAR, step interval to pickup

import mpi4py
from ase import io, parallel
from six.moves import cPickle as pickle
import numpy as np
import sys
import ANNcalculator as ac
import argparse

def OneForcefromOUTCAR():
    global OUT, out_idx, OUTlen
    t_force = []
    ftag = False; endtag = 0
    for i in range(out_idx, OUTlen):
        line = OUT[i]
        if 'TOTAL-FORCE' in line:
            ftag = True
        elif ftag == True:
            if '---' in line:
                endtag += 1
            else:
                tmp = line.replace('\n','').strip().split()[3:]
                t_force.append([float(tmp[0]), float(tmp[1]), float(tmp[2])])
        if endtag > 1:
            break

    out_idx = i + 1
    return np.array(t_force).astype(np.float32)

def OneEnergyfromOUTCAR():
    global OUT, out_idx, OUTlen
    for i in range(out_idx, OUTlen):
        line = OUT[i]
        if 'free  ' in line:
            energy = float(line.replace('\n','').strip().split()[-2])
            break

    out_idx = i + 1
    return energy

def save_data(file_name, atoms, calculator, atom_types, energy, force=None, calc_deriv=False):
    fdict = dict()
    symmax = dict()
    symmin = dict()
    fdict['energy'] = energy
    fdict['sym'] = dict()
    fdict['num'] = dict()
    if calc_deriv:
        fdict['force'] = force
        fdict['dsym'] = dict()

    for item in atom_types:
        type_idx = np.array(atoms.get_chemical_symbols()) == item
        symnum = acalc.symf_list.shape[1]
        fdict['sym'][item] = acalc.symf_list[type_idx,:]
        fdict['num'][item] = np.sum(type_idx)
        if calc_deriv:
            fdict['dsym'][item] = \
                acalc.dsymf_list[type_idx,:,:,:]

    with open(file_name, 'w') as fil:
        pickle.dump(fdict, fil, pickle.HIGHEST_PROTOCOL)

parser = argparse.ArgumentParser()
parser.add_argument('atom_types', \
                    help="Atom type seperated by ','. ex) 'Si,O' or 'Si,O,H' etc")
parser.add_argument('-d', '--calc_deriv', default=False, action='store_true', \
                    help="Whether to calculate the derivative of symmetry function. \
                        DEFAULT=False")
parser.add_argument('-c', '--coeffs_file', default='inp_fsymf',
                    help="text file which include coefficients for symmetry function. \
                        DEFAULT:inp_fsymf")
parser.add_argument('-i', '--input_file', default='input_datgen',
                    help="text file which include the directory of VASP results. \
                        DEFAULT:input_datgen")
argv = parser.parse_args()

comm = mpi4py.MPI.COMM_WORLD

with open(argv.input_file, 'r') as fil:
    inputs = fil.readlines()

xsf_idx = 0
acalc = ac.ANNpotential()
acalc.load_coeffs(argv.coeffs_file)

atom_types = argv.atom_types.split(',')
max_atom = dict()
for item in atom_types:
    max_atom[item] = 0

for item in inputs:
    params = item.replace('\n', '').strip().split()

    if params[1] == 'XDATCAR':
        if comm.rank == 0:
            with open(params[0] + '/XDATCAR', 'r') as file_xdat, open(params[0] + '/OUTCAR', 'r') as file_out:
                XDAT = file_xdat.readlines()
                OUT = file_out.readlines()
            out_idx = 0
            OUTlen = len(OUT)

            CONT_header = ''
            for _ in range(6):
                CONT_header += XDAT.pop(0)
 
            num_line = XDAT.pop(0)
            st_atnum = np.sum(np.array(num_line.replace('\n','').strip().split()).astype(np.int))
            CONT_header += num_line
            CONT_header += 'Direct\n'
 
        for mdstep in range(int(params[2])):
            if comm.rank == 0:
                XDAT.pop(0)
                at_posi = ''
                for i in range(st_atnum):
                    at_posi += XDAT.pop(0)
 
                force = OneForcefromOUTCAR()
                energy = OneEnergyfromOUTCAR()
 
            if mdstep % int(params[3]) == 0:
                xsf_idx += 1
                file_name = 'structure'+str(xsf_idx)+'.pickle'

                if comm.rank == 0:
                    with open('CONTCAR_tmp', 'w') as CONT:
                        CONT.write(CONT_header + at_posi)
 
                atoms = io.read('CONTCAR_tmp')
                acalc.get_symfunc(atoms, argv.calc_deriv)
                
                if comm.rank == 0: 
                    save_data(file_name, atoms, acalc, atom_types, \
                                energy, force, argv.calc_deriv)

                    print file_name, params[0], params[1], mdstep+1
                        
    else:
        xsf_idx += 1
        file_name = 'structure'+str(xsf_idx)+'.pickle'
        atoms = io.read(params[0]+'/CONTCAR')

        if comm.rank == 0:
            with open(params[0] + '/OUTCAR', 'r') as file_out:
                OUT = file_out.readlines()
            out_idx = 0
            OUTlen = len(OUT)

            force = OneForcefromOUTCAR()
            energy = OneEnergyfromOUTCAR()

        acalc.get_symfunc(atoms, argv.calc_deriv)
        if comm.rank == 0:
            save_data(file_name, atoms, acalc, atom_types, \
                        energy, force, argv.calc_deriv)            

            print file_name, params[0], params[1]

