import numpy as np
from six.moves import cPickle as pickle
from operator import itemgetter
import tensorflow as tf

def _get_dataset(filename_queue, atom_types, use_force=False, scale=None):
    mm_scale = 1.0

    DATA = dict()
    DATA['energy'] = list()
    DATA['sym'] = dict()
    tmp_sym = dict()
    DATA['num'] = dict()
    DATA['seg'] = dict()
    if use_force:
        DATA['force'] = list()
        DATA['dsym'] = dict()
        tmp_dsym = dict()

    for item in atom_types:
        DATA['sym'][item] = list()
        tmp_sym[item] = list()
        DATA['num'][item] = list()
        DATA['seg'][item] = list()
        if use_force:
            DATA['dsym'][item] = list()
            tmp_dsym[item] = list()

    for i,fname in enumerate(filename_queue):
        with open(fname, 'r') as fil:
            tmp_data = pickle.load(fil)

        DATA['energy'].append(tmp_data['energy'])
        if use_force:
            DATA['force'].append(tmp_data['force'])

        for item in atom_types:
            tmp_sym[item].append(tmp_data['sym'][item])
            DATA['num'][item].append(tmp_data['num'][item])
            DATA['seg'][item] += [i]*tmp_data['num'][item]
            if use_force:
                tmp_dsym[item].append(tmp_data['dsym'][item])

    atom_sum = np.sum(DATA['num'].values(), axis=0)
    max_atom = np.amax(atom_sum)
    DATA['energy'] = np.array(DATA['energy']).astype(np.float32)

    if use_force:
        DATA['force'] = np.concatenate(DATA['force'], axis=0).astype(np.float32)
        DATA['partition'] = np.ones([max_atom*len(DATA['energy'])]).astype(np.int)
        for i,item in enumerate(atom_sum):
            DATA['partition'][i*max_atom:i*max_atom+item] = 0

    for item in atom_types:
        DATA['sym'][item] = np.concatenate(tmp_sym[item], axis=0).astype(np.float32)
        DATA['num'][item] = np.array(DATA['num'][item])
        DATA['seg'][item] = np.array(DATA['seg'][item])

    if scale == None:
        scale_dict = dict()
        scale_dict['sym'] = dict()
        scale_dict['energy'] = np.zeros([2]) # min, max
        scale_dict['energy'][0] = np.amin(DATA['energy'])
        scale_dict['energy'][1] = np.amax(DATA['energy'])
        for item in atom_types:
            scale_dict['sym'][item] = np.zeros([2, DATA['sym'][item].shape[1]])
            scale_dict['sym'][item][0,:] = np.amin(DATA['sym'][item], axis=0)
            scale_dict['sym'][item][1,:] = np.amax(DATA['sym'][item], axis=0)

        with open('SCALE_FACTOR', 'w') as fil:
            pickle.dump(scale_dict, fil, pickle.HIGHEST_PROTOCOL)
    else:
        with open('SCALE_FACTOR', 'r') as fil:
            scale_dict = pickle.load(fil)

    # scaling process
    #DATA['energy'] = -mm_scale + 2.*mm_scale*(DATA['energy'] - scale_dict['energy'][0]) / \
    #                    (scale_dict['energy'][1] - scale_dict['energy'][0])
    #if use_force:
    #    DATA['force'] = 2.*mm_scale*DATA['force'] / \
    #                    (scale_dict['energy'][1] - scale_dict['energy'][0])
    
    for item in atom_types:
        val_range = scale_dict['sym'][item][1:2,:] - scale_dict['sym'][item][0:1,:]
        DATA['sym'][item] = -mm_scale + 2.*mm_scale*(DATA['sym'][item] - scale_dict['sym'][item][0:1,:]) / \
                                val_range
        
        if use_force:
            for j,jtem in enumerate(tmp_dsym[item]):
                tmp_shape = list(jtem.shape)
                tmp_shape[2] = max_atom - tmp_shape[2]
                tmp_dsym[item][j] = np.concatenate([jtem, np.zeros(tmp_shape)], axis=2)

            DATA['dsym'][item] = 2.*mm_scale*np.concatenate(tmp_dsym[item], axis=0) / \
                                val_range.reshape([1,tmp_shape[1],1,1]).astype(np.float32)

    return DATA, 2.*mm_scale/(scale_dict['energy'][1] - scale_dict['energy'][0])

def _get_batch(dataset, atom_types, batch_size=128, use_force=False):
    # do not use this function for full batch
    total_size = len(dataset['energy'])
    batch_idx = np.random.choice(total_size, batch_size, replace=False)

    batch = dict()
    batch['sym'] = dict()
    batch['num'] = dict()
    batch['seg'] = dict()
    if use_force:
        batch['dsym'] = dict()

    for item in atom_types:
        batch['sym'][item] = list()
        batch['seg'][item] = list()
        if use_force:
            batch['dsym'][item] = list()

    #batch['energy'] = map(dataset['energy'].__getitem__, batch_idx)
    batch['energy'] = dataset['energy'][batch_idx]
    atom_sum = np.sum(dataset['num'].values(), axis=0)
    max_atom = np.amax(atom_sum)
    atom_partsum = atom_sum[batch_idx]

    if use_force:
        force_idx = list()
        batch['partition'] = np.ones([max_atom*batch_size])
        in_batch = 0
        for i,inum in atom_sum:
            if i in batch_idx:
                batch['partition'][in_batch*max_atom:in_batch*max_atom+inum] = 0
                force_idx += [True]*inum
                in_batch += 1
            else:
                force_idx += [False]*inum
        batch['force'] = dataset['force'][force_idx]

    for item in atom_types:
        batch['num'][item] = dataset['num'][item][batch_idx]
        sym_idx = list()
        batch['seg'][item] = list()
        in_batch = 0
        for i,inum in enumerate(dataset['num'][item]):
            if i in batch_idx:
                sym_idx += [True]*inum
                batch['seg'][item] += [in_batch]*inum
                in_batch += 1
            else:
                sym_idx += [False]*inum
        batch['sym'][item] = dataset['sym'][item][sym_idx]
        batch['seg'][item] = np.array(batch['seg'][item])

        if use_force:
            batch['dsym'][item] = dataset['dsym'][item][sym_idx,:,:,:]

    return batch
