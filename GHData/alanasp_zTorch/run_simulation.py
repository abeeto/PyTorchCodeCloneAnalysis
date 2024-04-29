import sys
import multiprocessing as multiproc
import custom_logger
import datetime

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import pearsonr

import numpy as np

import ztorch_simulation as zsim


if __name__ == '__main__':

    start_time = datetime.datetime.utcnow()

    logger = custom_logger.get_logger('Run_Simulations')
    logger.info('Starting simulations...')

    num_time_steps = 10000
    if len(sys.argv) > 1:
        num_time_steps = int(sys.argv[1])

    on_the_fly = True
    if num_time_steps is not None and len(sys.argv) > 2:
        on_the_fly = bool(sys.argv[2])

    # (std, num_vnf_profiles, num_time_steps, output_file_prefix, input_file_prefix)
    params = [
        #{
        #    'std': 0.50,
        #    'num_init_profiles': 100,
        #    'steps': num_time_steps,
        #    'input_file': not on_the_fly,
        #    'on_the_fly': on_the_fly
        #},
        {
            'std': 0.10,
            'num_init_profiles': 1000,
            'steps': num_time_steps,
            'input_file': not on_the_fly,
            'on_the_fly': on_the_fly
        },
        #{
        #    'std': 0.50,
        #    'num_init_profiles': 1000,
        #    'steps': num_time_steps,
        #    'input_file': not on_the_fly,
        #    'on_the_fly': on_the_fly
        #},
        {
            'std': 0.06,
            'num_init_profiles': 1000,
            'steps': num_time_steps,
            'input_file': not on_the_fly,
            'on_the_fly': on_the_fly
        },
        {
            'std': 0.08,
            'num_init_profiles': 1000,
            'steps': num_time_steps,
            'input_file': not on_the_fly,
            'on_the_fly': on_the_fly
        },
        {
            'std': 0.12,
            'num_init_profiles': 1000,
            'steps': num_time_steps,
            'input_file': not on_the_fly,
            'on_the_fly': on_the_fly
        },
    ]

    consumption_params = {
        'std': [0.01, 0.10, 0.3],
        'num_init_profiles': 1000,
        'steps': num_time_steps,
        'input_file': False,
        'on_the_fly': True
    }

    unified_time = np.linspace(0, num_time_steps, num=50)

    measure_mon_data_consumption = True
    if measure_mon_data_consumption:
        num_runs = 20
        interp_groups_non_varied = np.zeros(unified_time.shape)
        interp_groups_varied = np.zeros(unified_time.shape)

        consumption_non_varied = 0
        consumption_varied = 0
        for i in range(num_runs):
            consumption_params['varied_mon_freq'] = False
            sim = zsim.Simulation(**consumption_params)
            consumption, aff_groups = sim.run_sim()
            aff_groups = np.array(aff_groups)
            time, vals = aff_groups[:, 0], aff_groups[:, 1]
            interp_groups_non_varied += interp1d(time, vals)(unified_time)
            consumption_non_varied += consumption
            print('Non-varied: {}'.format(consumption))
            print(aff_groups)

            consumption_params['varied_mon_freq'] = True
            sim = zsim.Simulation(**consumption_params)
            consumption, aff_groups = sim.run_sim()
            aff_groups = np.array(aff_groups)
            time, vals = aff_groups[:, 0], aff_groups[:, 1]
            interp_groups_varied += interp1d(time, vals)(unified_time)
            consumption_varied += consumption
            print('Varied: {}'.format(consumption))
            print(aff_groups)
        #calculate averages
        interp_groups_non_varied /= num_runs
        interp_groups_varied /= num_runs
        consumption_non_varied /= num_runs
        consumption_varied /= num_runs

        print(interp_groups_non_varied)
        print(interp_groups_varied)
        print('Avg consumption non-varied: {}'.format(consumption_non_varied))
        print('Avg consumption varied: {}'.format(consumption_varied))
        std = consumption_params['std']
        num_prof = consumption_params['num_init_profiles']
        filename = 'results/varied_v_non_varied_{}_{}_{}_{}'.format(int(100*std[0]), int(100*std[1]),
                                                                           int(100*std[2]), num_prof)
        with open(filename, 'w') as file:
            li_time = np.ndarray.tolist(unified_time)
            li_non_var = np.ndarray.tolist(interp_groups_non_varied)
            li_var = np.ndarray.tolist(interp_groups_varied)

            str_time = ' '.join(map(str, li_time)) + '\n'
            str_non_var = ' '.join(map(str, li_non_var)) + '\n'
            str_var = ' '.join(map(str, li_var)) + '\n'

            file.write(str(consumption_non_varied) + '\n')
            file.write(str(consumption_varied) + '\n')
            file.write(str_time)
            file.write(str_non_var)
            file.write(str_var)
        exit(0)

    procs = []

    for param in params:
        sim = zsim.Simulation(**param)
        proc = multiproc.Process(target=sim.run_sim)
        proc.start()
        procs.append(proc)

    for proc in procs:
        proc.join()

    end_time = datetime.datetime.utcnow()
    delta = int((end_time - start_time).seconds)
    logger.info('Simulations finished in {}h {}min {}s'.format(delta//3600, (delta % 3600)//60, delta % 60))
