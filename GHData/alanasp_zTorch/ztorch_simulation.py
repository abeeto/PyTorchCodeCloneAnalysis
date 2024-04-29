import numpy as np
import custom_logger
import pathlib
import os

from timeseries import TimeSeries, TSEntry
import utils

base_vnf_profiles = {
     'low':  {'MME':  [17.7, 15.9, 5.8],
              'SGW':  [0.7, 0.3, 0.14],
              'HSS':  [0.9, 1.1, 0.7],
              'PCRF': [1.2, 0.6, 0.5],
              'PGW':  [1.7, 2.1, 0.8]},

     'high': {'MME':  [2.9, 3.8, 1.9],
              'SGW':  [79.1, 3.3, 91.2],
              'HSS':  [2.9, 4.5, 1.3],
              'PCRF': [1.9, 3.9, 0.9],
              'PGW':  [53.1, 37.2, 92]}
}

default_params = {
    'surv_epoch': 500,
    'min_surv_delta': 10,
    'min_surv_epoch': 500,
    'mon_periods': [2, 5, 10, 20, 50],
    'default_mon_period_id': 2,
    'learning_rate': 0.5,
    'discount_rate': 0.9,
    'random_action_prob': 0.5
}


class Simulation:
    # Generates time series for each vnf
    # std can be a single number or a list of size 3, which will indicate different stds for different features
    # If output_file is True, writes time series to the default file (can specify custom file base name)
    # If input_file is True, reads time series from the default file (can specify custom file base name)
    # The custom specified name will be appended with _data for time series data, so specifying file name XYZ
    #   means that the function will expect a file XYZ_data
    # on_the_fly option if set to True generates data as needed w/o writing to or reading from file and
    #   w/o keeping it in memory
    def __init__(self, std=0.1, num_init_profiles=1000, steps=None, output_file=None, input_file=None,
                 results_dir=True, on_the_fly=False, varied_mon_freq=False):
        if type(std) is list:
            self.logger = custom_logger.get_logger('Simulation_{}_{}_{}_{}'.format(int(std[0]*100), int(std[1]*100),
                                                                                   int(std[2]*100), num_init_profiles))
        else:
            self.logger = custom_logger.get_logger('Simulation_{}_{}'.format(int(std*100), num_init_profiles))

        self.logger.info('Initialising simulation...')

        self.std = std
        self.num_profiles = num_init_profiles

        self.total_steps = steps

        self.varied_mon_freq = varied_mon_freq

        if results_dir is True:
            # create results directory
            results_dir = 'results/'
            pathlib.Path(results_dir).mkdir(parents=True, exist_ok=True)
        self.results_dir = results_dir

        # read time series data from file
        if input_file:

            self.logger.info('Reading time series from files...')

            if input_file is True:
                if type(std) is list:
                    input_file = 'data/data_{}_{}_{}_{}/ztorch_out'.format(int(std[0]*100), int(std[1]*100),
                                                                           int(std[2]*100), num_init_profiles)
                else:
                    input_file = 'data/data_{}_{}/ztorch_out'.format(int(std * 100), num_init_profiles)

            # read init profiles into memory, next items will be read during simulation execution
            with open(input_file + '0', 'rb') as data_file:
                num_vnf_profiles = int(data_file.readline().decode('UTF-8').split(' ')[1])
                num_time_steps = int(data_file.readline().decode('UTF-8').split(' ')[1])
                if self.total_steps is None:
                    self.total_steps = num_time_steps
                else:
                    self.total_steps = min(self.total_steps, num_time_steps)

                init_profiles = np.reshape(np.fromstring(data_file.read()), (-1, 3))
                self.time_series = TimeSeries(init_profiles, file_prefix=input_file, max_time=num_time_steps)

        # generate time series either to file or to memory
        elif output_file is not None or not on_the_fly:
            if self.total_steps is None:
                self.total_steps = 10000

            self.logger.info('Generating time series...')

            init_profiles = utils.gen_init_profiles(base_vnf_profiles['high'],
                                              num_init_profiles//len(base_vnf_profiles['high']))

            if output_file is True:
                if type(std) is list:
                    output_file = 'data/data_{}_{}_{}_{}/ztorch_out'.format(int(std[0]*100), int(std[1]*100),
                                                                            int(std[2]*100), num_init_profiles)
                else:
                    output_file = 'data/data_{}_{}/ztorch_out'.format(int(std * 100), num_init_profiles)
                pathlib.Path(output_file).mkdir(parents=True, exist_ok=True)
                output_file += '/ztorch_out'

            data_file = None
            if output_file:
                with open(output_file + '0', 'wb') as data_file:
                    data_file.write(bytes('num_vnf_profiles {}\n'.format(len(init_profiles)), encoding='UTF-8'))
                    data_file.write(bytes('num_time_steps {}\n'.format(steps), encoding='UTF-8'))
                    data_file.write(init_profiles.tostring())

            # time series of each vnf compute needs
            if output_file:
                self.time_series = TimeSeries(init_profiles, file_prefix=output_file, max_time=self.total_steps)
            else:
                self.time_series = TimeSeries(init_profiles)

            for step in range(1, self.total_steps+1):
                if step % 1000 == 0:
                    self.logger.info('Generating {}th step of time series...'.format(step))
                delta = np.random.normal(0, std, self.time_series.shape)
                new_entry = TSEntry(init_profiles+delta, step)
                self.time_series.add_entry(new_entry)

            self.time_series.load_entry(time=0)

        elif on_the_fly and steps is not None:
            if self.total_steps is None:
                self.total_steps = 10000

            self.logger.info('Setting up ON THE FLY time series...')

            init_profiles = utils.gen_init_profiles(base_vnf_profiles['high'],
                                                    num_init_profiles // len(base_vnf_profiles['high']))

            self.time_series = TimeSeries(init_profiles, on_the_fly=True, std=self.std, max_time=self.total_steps)

        else:
            raise Exception('Invalid options!')

        self.logger.info('Time series initialized!')

        # create default gravity centres for ekm based on base vnf profiles
        self.default_centres = list()
        for key in base_vnf_profiles['high']:
            self.default_centres.append(base_vnf_profiles['high'][key])
        self.default_centres = np.array(self.default_centres)

        # centres evolution will contain evolution of centres of gravity after running k-means
        self.centres_evolution = list()

        # Q-Learning table, to be filled during first run of simulation and then further updated
        self.q_table = np.zeros((len(init_profiles)+1, 5))

        # attempt loading trained q_table from file
        self.load_q_table()

        # Number of times each Q-Table state was visited
        self.num_visited = np.zeros(len(init_profiles)+1)

        self.logger.info('Simulation initialised!')

    def run_sim(self, centres=None, params=default_params):
        surv_epoch = params['surv_epoch']
        mon_periods = params['mon_periods']
        mon_id = params['default_mon_period_id']
        min_surv_delta = params['min_surv_delta']
        min_surv_epoch = params['min_surv_epoch']
        self.logger.info('Running simulation with surveillance epoch of {}t...'.format(surv_epoch))
        if centres is None:
            centres = np.array([[75, 75, 75], [25, 25, 25]])

        ts_entry = self.time_series.current

        steps, centres, aff_groups, points, granularity = self.run_ekm(init_centres=centres, points=ts_entry.entry)

        last_aff_adjustment = 0
        old_aff_groups = aff_groups
        old_centres = centres
        prev_points = points

        num_deviations = list()
        alerts = [0]

        num_aff_groups = [(0, len(centres))]
        mon_indices = [(0, mon_id+1)]
        surv_epoch_lengths = [(0, surv_epoch)]

        last_surv_time = 0
        last_mon_time = 0

        mon_data_consumed = 0

        #only used with varied_mon_freq
        feature_mon_ids = np.array([mon_id]*points.shape[1])
        last_feature_mon_times = np.array([0]*points.shape[1])
        last_observed_vals = np.array(points)

        while ts_entry:
            points = ts_entry.entry

            # conduct monitoring
            if ts_entry.time - last_mon_time >= mon_periods[mon_id]:
                if self.varied_mon_freq:
                    #conduct monitoring by feature if mon_period elapsed
                    for i in range(len(last_feature_mon_times)):
                        if ts_entry.time - last_feature_mon_times[i] > mon_periods[feature_mon_ids[i]]:
                            last_observed_vals[:, i] = points[:, i]
                            last_feature_mon_times[i] = ts_entry.time
                            mon_data_consumed += points.shape[0]
                    last_mon_time = ts_entry.time
                    #count deviations with the values that we actually observed
                    alerts[-1] += utils.count_deviations(last_observed_vals, aff_groups, centres, granularity)
                else:
                    mon_data_consumed += points.shape[0] * points.shape[1]
                    last_mon_time = ts_entry.time
                    alerts[-1] += utils.count_deviations(points, aff_groups, centres, granularity)

            # end of surveillance epoch
            if ts_entry.time - last_surv_time >= surv_epoch:

                num_aff_groups.append((ts_entry.time, len(centres)))
                mon_indices.append((ts_entry.time, mon_id+1))
                surv_epoch_lengths.append((ts_entry.time, surv_epoch))

                reprofile = False

                alerts[-1] = min(alerts[-1], points.shape[0])
                # check alerts and ask for reprofiling if there were deviations in 2 consecutive periods
                if alerts[-1] > 0 and (len(alerts) < 2 or alerts[-2] == 0):
                    mon_id = max(0, mon_id-1)
                elif alerts[-1] > 0 and alerts[-2] > 0:
                    # reset monitoring frequency
                    mon_id = params['default_mon_period_id']
                    reprofile = True
                elif alerts[-1] == 0:
                    mon_id = min(len(mon_periods)-1, mon_id+1)

                if self.varied_mon_freq:
                    feature_mon_ids = np.array([mon_id] * points.shape[1])
                    mon_period = mon_periods[mon_id]
                    max_std = max(self.std)
                    if type(self.std) is list:
                        for i in range(len(self.std)):
                            ratio = max_std/self.std[i]
                            this_period = mon_period/ratio
                            for j in range(len(mon_periods)):
                                if mon_periods[j] >= this_period:
                                    feature_mon_ids[i] = j

                devs = utils.count_deviations(points, aff_groups, centres, granularity)
                num_deviations.append(devs)

                if len(alerts) >= 2:
                    # action taken is the difference between epoch lengths adjusted to non-negative range
                    action_taken = self.q_table.shape[1]//2 + \
                                   (surv_epoch_lengths[-1][1] - surv_epoch_lengths[-2][1])//min_surv_delta
                    # we were in a state 'alerts[-2]', took action and ended up in 'alerts[-1]'
                    self.update_q_table(alerts[-2], action_taken, alerts[-1], surv_epoch_lengths[-1][1])

                action = self.get_action(alerts[-1])
                surv_epoch += min_surv_delta*(action - self.q_table.shape[1]//2)
                surv_epoch = max(surv_epoch, min_surv_epoch)
                implied_std = np.sqrt(np.average(np.square(points-self.time_series.first.entry)))
                self.logger.info('Time: {} Surv epoch: {} Mon_id: {} Alerts: {} Implied std: {} std: {} Aff groups: {} Action: {} '.
                                 format(ts_entry.time, surv_epoch, mon_id, alerts[-1], implied_std, self.time_series.std, len(centres), action))

                if len(alerts) > 1 and alerts[-1] > 0 and alerts[-2] > 0 \
                        and last_aff_adjustment < last_surv_time:
                    reprofile = True
                    last_aff_adjustment = ts_entry.time
                    # 2 affinity groups is the lower bound
                    if len(centres) > 2:
                        centres = centres[:-1]
                        self.logger.info('Decreasing affinity groups to {}'.format(len(centres)))

                # no deviation occured for 2 consecutive epochs
                elif len(alerts) > 1 and alerts[-1] == 0 and alerts[-2] == 0 \
                        and last_aff_adjustment < last_surv_time:
                    # pick a random vnf profile to act as a new centre
                    cid = np.random.randint(len(ts_entry.entry))
                    centres = np.append(centres, [ts_entry.entry[cid]], axis=0)
                    self.logger.info('Increasing affinity groups to {} with a new center at {}'.
                                     format(len(centres), centres[-1]))

                    # rerun ekm with new centres
                    reprofile = True
                    last_aff_adjustment = ts_entry.time

                old_aff_groups = aff_groups
                old_centres = centres
                if reprofile:
                    steps, centres, aff_groups, points, granularity = self.run_ekm(init_centres=centres, points=ts_entry.entry)

                prev_points = points
                alerts.append(0)
                last_surv_time = ts_entry.time

            ts_entry = self.time_series.get_next(delta=mon_periods[mon_id])

        num_aff_groups.append((self.total_steps, len(centres)))
        mon_indices.append((self.total_steps, mon_id + 1))
        surv_epoch_lengths.append((self.total_steps, surv_epoch))

        # write results to file for further processing
        if self.results_dir:
            tuple_to_str = lambda t: str(t[0]) + ' ' + str(t[1])

            if type(self.std) is list:
                file_name = self.results_dir + 'num_aff_groups_{}_{}_{}_{}'.format(int(self.std[0] * 100),
                                                                                   int(self.std[1] * 100),
                                                                                   int(self.std[2] * 100),
                                                                                   self.num_profiles)
            else:
                file_name = self.results_dir + 'num_aff_groups_{}_{}'.format(int(100 * self.std), self.num_profiles)

            with open(file_name, 'w') as data_file:
                data_file.write(str(len(num_aff_groups)) + '\n')
                data_file.write('\n'.join(map(tuple_to_str, num_aff_groups)))

            if type(self.std) is list:
                file_name = self.results_dir + 'mon_indices_{}_{}_{}_{}'.format(int(self.std[0] * 100),
                                                                                int(self.std[1] * 100),
                                                                                int(self.std[2] * 100),
                                                                                self.num_profiles)
            else:
                file_name = self.results_dir + 'mon_indices_{}_{}'.format(int(100 * self.std), self.num_profiles)

            with open(file_name, 'w') as data_file:
                data_file.write(str(len(mon_indices)) + '\n')
                data_file.write('\n'.join(map(tuple_to_str, mon_indices)))

            if type(self.std) is list:
                file_name = self.results_dir + 'surv_epoch_lengths_{}_{}_{}_{}'.format(int(self.std[0] * 100),
                                                                                       int(self.std[1] * 100),
                                                                                       int(self.std[2] * 100),
                                                                                       self.num_profiles)
            else:
                file_name = self.results_dir + 'surv_epoch_lengths_{}_{}'.format(int(100 * self.std), self.num_profiles)

            with open(file_name, 'w') as data_file:
                data_file.write(str(len(surv_epoch_lengths)) + '\n')
                data_file.write('\n'.join(map(tuple_to_str, surv_epoch_lengths)))

        self.write_q_table()
        self.logger.info('Simulation finished!')
        self.logger.info('FINAL STATS Number of affinity groups: {}'.format(len(centres)))
        return mon_data_consumed, num_aff_groups

    def load_q_table(self):

        if type(self.std) is list:
            q_filename = 'config/q_table_{}_{}_{}_{}'.format(int(self.std[0] * 100), int(self.std[1] * 100),
                                                                        int(self.std[2] * 100), self.num_profiles)
        else:
            q_filename = 'config/q_table_{}_{}'.format(int(self.std * 100), self.num_profiles)
        if os.path.exists(q_filename):
            with open(q_filename, 'rb') as q_file:
                self.q_table = np.reshape(np.fromstring(q_file.read()), self.q_table.shape)
        else:
            incr = 500.0
            # fill q-table with reasonable initial values
            for j in range(self.q_table.shape[1]):
                self.q_table[0][j] = incr * j

            for i in range(1, self.q_table.shape[0]):
                for j in range(self.q_table.shape[1]):
                    mult = (self.q_table.shape[1]//2 - j)/(self.q_table.shape[1]//2)
                    self.q_table[i][j] = self.q_table[i-1][j] + mult*incr

    def write_q_table(self):
        pathlib.Path('config').mkdir(parents=True, exist_ok=True)
        if type(self.std) is list:
            q_filename = 'config/q_table_{}_{}_{}_{}'.format(int(self.std[0] * 100), int(self.std[1] * 100),
                                                                        int(self.std[2] * 100), self.num_profiles)
        else:
            q_filename = 'config/q_table_{}_{}'.format(int(self.std * 100), self.num_profiles)
        with open(q_filename, 'wb') as q_file:
            q_file.write(self.q_table.tostring())

    # runs enhanced k-means clustering algorithm
    def run_ekm(self, init_centres=None, points=None):
        if init_centres is None:
            init_centres = self.default_centres
        if points is None:
            points = self.time_series.first.entry
        self.logger.info('Running ekm with {} clusters and {} points...'.format(len(init_centres), len(points)))
        centres = np.array(init_centres)
        steps = 0
        aff_groups = [-1] * len(points)
        converged = False
        granularity = 1e-100
        while not converged:
            power = min(len(points), 1000)
            granularity = max(granularity, 100 * steps ** (np.sqrt(power)) / 2.0 ** power)
            for i in range(len(points)):
                min_dist = 1e10
                for j in range(len(centres)):
                    dist = np.linalg.norm(points[i] - centres[j])
                    if dist < min_dist:
                        min_dist = dist
                        aff_groups[i] = j
            new_centres = self.calc_centres(points, aff_groups, len(centres))

            centres = self.snap_to_grid(centres, granularity)
            new_centres = self.snap_to_grid(new_centres, granularity)

            if np.all(np.equal(new_centres, centres)):
                converged = True
            centres = new_centres
            steps += 1
        self.logger.info('ekm converged in {} steps'.format(steps))
        return steps, centres, aff_groups, points, granularity

    def get_action(self, state, random_prob=0.5):
        is_random = (np.random.uniform(0.0, 1.0) < random_prob)
        # return random action (for exploration purposes)
        if is_random:
            return np.random.randint(0, self.q_table.shape[1])
        # otherwise return the best action
        best_action = 0
        for action in range(len(self.q_table[state])):
            if self.q_table[state, action] > self.q_table[state, best_action]:
                best_action = action
        return best_action

    def get_reward(self, num_deviations, surv_epoch_length, beta=0.5):
        # ensure reward function is able to deal with 0 deviations
        if num_deviations == 0:
            num_deviations = 0.5
        return surv_epoch_length/(num_deviations**beta)

    def update_q_table(self, state, action, next_state, surv_epoch_length, learning_rate=0.5, discount_rate=0.9):
        q_max = 0
        # loop through the actions in next state to find max reward
        for reward in self.q_table[next_state]:
            q_max = max(q_max, reward)

        self.q_table[state, action] = (1-learning_rate)*self.q_table[state, action] + \
            learning_rate*(self.get_reward(state, surv_epoch_length) + discount_rate*q_max)

    def calc_centres(self, points, point_group, num_centres):
        counts = [0]*num_centres
        centres = [[0]*len(points[0]) for _ in range(num_centres)]
        for pid in range(len(points)):
            pg = point_group[pid]
            counts[pg] += 1
            centres[pg] += points[pid]
        for cid in range(len(centres)):
            if counts[cid] > 0:
                centres[cid] /= counts[cid]
        centres = np.array(centres)
        return np.array(centres)

    def snap_to_grid(self, points, grid_step):
        grid_points = list()
        for point in points:
            gp = list()
            for coord in point:
                gp.append(min(100.0, np.round(coord/grid_step)*grid_step))
            grid_points.append(gp)
        return np.array(grid_points)
