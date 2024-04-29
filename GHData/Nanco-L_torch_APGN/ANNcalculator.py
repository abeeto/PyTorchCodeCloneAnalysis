"""ASE calculator to use ANN potential"""

import sys
from ase.parallel import parprint
from ase import neighborlist
import ase.calculators.interface as asecif
import numpy as np
from six.moves import cPickle as pickle
import symfunc as sf
import itertools
import timeit

class ANNpotential(asecif.Calculator):
    """ASE calculator.

    A calculator should store a copy of the atoms object used for the
    last calculation.  When one of the *get_potential_energy*,
    *get_forces*, or *get_stress* methods is called, the calculator
    should check if anything has changed since the last calculation
    and only do the calculation if it's really needed.  Two sets of
    atoms are considered identical if they have the same positions,
    atomic numbers, unit cell and periodic boundary conditions."""

    def __init__(self):
        # Dongsun: get MPI variables from ase.parallel module if mpi4py exists.
        if "mpi4py" in sys.modules:
            module = sys.modules["ase.parallel"]
            self.parallel = True
            self.rank = module.rank
            self.size = module.size
            self.barrier = module.barrier
            self.comm = module.world.comm
        else:
            self.parallel = False

        self.atom_types = None
        self.cutoff = None
        self.coeffs = None
        self.symf_list = None
        self.dsymf_list = None
        #self. # weights
 
    def load_weights(self):
        """
        Tensorflow part will be changed to numpy..(for speed?)
        """
        return 0

    def load_coeffs(self, file_name):
        with open(file_name, 'r') as fil:
            coeffs = fil.readlines()

        self.atom_types = coeffs[0].replace('\n','').split()
        self.cutoff = float(coeffs[1].replace('\n',''))
        self.coeffs = list()
        for line in coeffs[2:]:
            tmp = line.replace('\n','').split()
            self.coeffs.append({'type':tmp[0],\
                                'neighbors':tmp[1],\
                                'coeffs':np.array(tmp[2:]).astype(np.float)})

        self.neigh_types = list(set([item['neighbors'] for item in self.coeffs]))

    def load_scale(self, file_name):
        with open(file_name, 'r') as fil:
            self.scale = pickle.load(fil)

    def _get_symfunc_for_atom(self, center, atoms=None, calc_derivative=False):
        num_atom_types = len(self.atom_types)
        chem_symbols = np.array(atoms.get_chemical_symbols())
        sym_vec_leng = len(self.coeffs)
        
        nset = neighborlist.NeighborList([self.cutoff/2]*len(atoms), bothways=True, skin=0.0)
        nset.update(atoms)
        indices, offsets = nset.get_neighbors(center)

        remove_self_idx = indices == center
        for i in range(3):
            remove_self_idx = np.logical_and(remove_self_idx, offsets[:,i] == 0)
        remove_self_idx = np.logical_not(remove_self_idx)
        indices = indices[remove_self_idx]
        offsets = offsets[remove_self_idx]

        vectors = list()
        for i, offset in zip(indices, offsets):
            vectors.append(atoms.positions[i] + np.dot(offset, atoms.get_cell()) - \
                           atoms.positions[center])
        vectors = np.array(vectors)

        duo = dict()
        trio = dict()

        duo['dists'] = np.linalg.norm(vectors, axis=1, keepdims=True)
        duo['cutfunc'] = sf.cutofffunc(self.cutoff, duo['dists'])
        duo['indices'] = indices

        trio_combi = np.array(list(itertools.combinations(range(len(indices)), 2)))
        trio_far_vectors = vectors[trio_combi[:,1]] - vectors[trio_combi[:,0]]
        trio['dists'] = \
            np.concatenate((np.squeeze(duo['dists'][trio_combi], 2), \
                            np.linalg.norm(trio_far_vectors, axis=1, keepdims=True)), \
                            axis=1)
        trio['dists_sqsum'] = np.sum(trio['dists']**2, axis=1, keepdims=True)
        trio['cutfunc'] = sf.cutofffunc(self.cutoff, trio['dists'])
        trio['cos'] = (trio['dists'][:,0:1]**2 + trio['dists'][:,1:2]**2 - trio['dists'][:,2:3]**2) / \
                    2/trio['dists'][:,0:1]/trio['dists'][:,1:2]
        trio['indices'] = indices[trio_combi]
        
        if calc_derivative:
            duo['dists_deri'] = vectors / duo['dists']
            duo['dcutfunc'] = sf.dcutofffunc(self.cutoff, duo['dists'])

            trio['dists_deri'] = \
                np.concatenate((vectors[trio_combi[:,0]] / trio['dists'][:,0:1],\
                                vectors[trio_combi[:,1]] / trio['dists'][:,1:2],\
                                trio_far_vectors / trio['dists'][:,2:3]), \
                                axis=1)
            trio['dcutfunc'] = sf.dcutofffunc(self.cutoff, trio['dists'])
            trio['dcos'] = np.zeros([trio['dists'].shape[0], 3])
            trio['dcos'][:,0] = 0.5*(1/trio['dists'][:,1] + 1/(trio['dists'][:,0]**2) * \
                                (trio['dists'][:,2]**2/trio['dists'][:,1] - trio['dists'][:,1]))
            trio['dcos'][:,1] = 0.5*(1/trio['dists'][:,0] + 1/(trio['dists'][:,1]**2) * \
                                (trio['dists'][:,2]**2/trio['dists'][:,0] - trio['dists'][:,0]))
            trio['dcos'][:,2] = -trio['dists'][:,2]/trio['dists'][:,0]/trio['dists'][:,1] 

        prev_neitype = None
        for i, coeff in enumerate(self.coeffs):
            #print i, self.rank
            if coeff['neighbors'] != prev_neitype:
                prev_neitype = coeff['neighbors']
                tmp_types = prev_neitype.split('|')
                if tmp_types[1] == '--':
                    tmp_idxs = chem_symbols[indices] == tmp_types[0]
                else:
                    tmp_idxs = np.logical_or(np.logical_and(chem_symbols[trio['indices'][:,0]] == tmp_types[0], \
                                                            chem_symbols[trio['indices'][:,1]] == tmp_types[1]), \
                                             np.logical_and(chem_symbols[trio['indices'][:,0]] == tmp_types[1], \
                                                            chem_symbols[trio['indices'][:,1]] == tmp_types[0]))

            if coeff['type'] == '2':
                if np.sum(tmp_idxs) == 0:
                    break
                self.symf_list[center, i] = sf.symfunc2(coeff['coeffs'], tmp_idxs, duo)
                if calc_derivative:
                    self.dsymf_list[center,i,:,:] = sf.dsymfunc2(coeff['coeffs'], tmp_idxs, duo, center, len(atoms))
            elif coeff['type'] == '4':
                if np.sum(tmp_idxs) == 0:
                    break
                self.symf_list[center, i] = sf.symfunc4(coeff['coeffs'], tmp_idxs, trio)
                if calc_derivative:
                    self.dsymf_list[center,i,:,:] = sf.dsymfunc4(coeff['coeffs'], tmp_idxs, trio, center, len(atoms))
            else:
                print 'This type of symmetry function is not supported'

    def get_symfunc(self, atoms=None, calc_derivative=False, scale=False):
        time1 = timeit.default_timer()
        sym_vec_leng = len(self.coeffs)
 
        self.symf_list = np.zeros([len(atoms), sym_vec_leng])
        self.dsymf_list = np.zeros([len(atoms), sym_vec_leng, len(atoms), 3])
 
        # Dongsun: distribute the load (atoms) across the processes.
        if self.parallel:
            begins = []
            ends = []
            q = len(atoms) // self.size
            r = len(atoms) % self.size
            for rank in range(self.size):
                begin = rank * q + min(rank, r)
                end = begin + q
                if r > rank:
                    end += 1
                begins.append(begin)
                ends.append(end)
 
            begin = begins[self.rank]
            end = ends[self.rank]
        else:
            begin = 0
            end = len(atoms.positions)
 
        for a in range(begin, end):
            self._get_symfunc_for_atom(a, atoms, calc_derivative)

        if self.parallel:
            self.barrier()
            for i in range(len(atoms)):
                # Find the rank where i-th atom belongs to.
                for rank in range(self.size):
                    if i < ends[rank]:
                        break
            # Broadcast forces of i-th atom.
                self.symf_list[i,:] = self.comm.bcast(self.symf_list[i,:], root=rank)
                self.dsymf_list[i,:,:,:] = self.comm.bcast(self.dsymf_list[i,:,:,:], root=rank)
            self.barrier()

        if scale:
            self.scaleSymlist(calc_derivative) # After check this code, remove #

        self.symf_list = self.symf_list.astype(np.float32)
        self.dsymf_list = self.dsymf_list.astype(np.float32) 

    def scaleSymlist(self, calc_derivative=False):
        leng = int(self.sym.scale.shape[1])
        self.symf_list = -1. + 2.*(self.symf_list - self.sym.scale[0,:])/self.sym.scale[1,:]
        if calc_derivative:
            self.dsymf_list = 2.*self.dsymf_list / self.sym.scale[1,:].reshape([1,1,leng,1])
 
    def get_potential_energy(self, atoms=None, force_consistent=False):
        """Return total energy.
        
        Both the energy extrapolated to zero Kelvin and the energy
        consistent with the forces (the free energy) can be
        returned."""
        return 0
        
    def get_forces(self, atoms):
        """Return the forces."""
        return 0
            
    def get_stress(self, atoms):
        """Return the stress."""
        return 0
 
    def calculation_required(self, atoms, quantities):
        """Check if a calculation is required.
 
        Check if the quantities in the *quantities* list have already
        been calculated for the atomic configuration *atoms*.  The
        quantities can be one or more of: 'energy', 'forces', 'stress',
        'charges' and 'magmoms'.
        
        This method is used to check if a quantity is available without
        further calculations.  For this reason, calculators should 
        react to unknown/unsupported quantities by returning True,
        indicating that the quantity is *not* available."""
        return False 
