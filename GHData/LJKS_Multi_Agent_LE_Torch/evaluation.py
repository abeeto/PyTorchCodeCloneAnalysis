import argparse
import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm

def sort_key_log_steps(file_name):
    file_name = file_name.replace('log_step_', '')
    file_name = file_name.replace('.pickle', '')
    key = int(file_name)
    return key

def analyze_baseline_vs_tscl_success(runs, senders_receivers=[10,10], steps=400):
    tscl_runs = []
    baseline_runs = []
    for run in runs:
        run['hparams'] = vars(run['hparams'])
        if not (run['hparams']['num_senders'] == senders_receivers[0] and run['hparams']['num_receivers'] == senders_receivers[1]):
            continue
        if not run['num_steps'] >=steps:
            continue
        if run['hparams']['experiment'] == 'tscl_population_training':
            if 'tscl' in run['hparams']['run_key']:
                print('removed', run['hparams']['run_key'])
                continue
            print('adding to tscl', run['hparams']['run_key'])
            tscl_runs.append(run)
            print(run['hparams']['run_key'])
        elif run['hparams']['experiment'] == 'baseline_population_training':
            print('adding to baseline', run['hparams']['run_key'])

            baseline_runs.append(run)



    rows = []
    eval_keys = ['training_loss', 'training_acc', 'test_loss', 'test_acc']

    for step in range(steps):
        for tscl_run in tscl_runs:
            step_res_tscl = []
            step_data = tscl_run['res_steps'][step]
            for eval_key in eval_keys:
                step_res = np.mean([step_data[_key] for _key in step_data.keys() if eval_key in _key])
                step_res_tscl.append(step_res)
            row = [step, tscl_run['num_steps'], tscl_run['hparams']['experiment']] + step_res_tscl
            rows.append(row)

        for baseline_run in baseline_runs:
            step_res_baseline = []
            step_data = baseline_run['res_steps'][step]
            for eval_key in eval_keys:
                step_res = np.mean([step_data[_key] for _key in step_data.keys() if eval_key in _key])
                step_res_baseline.append(step_res)
            row = [step, baseline_run['num_steps'], baseline_run['hparams']['experiment']] + step_res_baseline
            rows.append(row)

        res_df = pd.DataFrame(columns = ['step', 'run_max_steps', 'experiment'] + eval_keys, data=rows)
    return res_df
    """
    sns.lineplot(data=res_df, x='step', y='test_acc', hue='experiment')
    plt.show()
    sns.lineplot(data=res_df, x='step', y='test_loss', hue='experiment')
    plt.show()
    sns.lineplot(data=res_df, x='step', y='training_acc', hue='experiment')
    plt.show()
    sns.lineplot(data=res_df, x='step', y='training_loss', hue='experiment')
    plt.show()
    print(res_df)
    """

def runs_dataframe(source='results'):
    dirs = os.listdir(source)
    res_directories = ['results/' + dir for dir in dirs if not dir=='rundocs']
    runs = [[res_dir + '/' + run_dir for run_dir in os.listdir(res_dir)] for res_dir in res_directories]
    run_res = []
    for run_type in runs:
        for setting_path in run_type:
            print(setting_path)
            for run_path in os.listdir(setting_path):
                #print(run_path)
                run_path = setting_path + '/' + run_path
                with open(run_path + '/hyperparameters.pickle', 'rb') as file:
                    hyperparams = pickle.load(file)
                res_dirs = os.listdir(run_path + '/results')
                res_dirs = [res_dir for res_dir in res_dirs if res_dir.startswith('log_step')]
                res_dirs = sorted(res_dirs, key=sort_key_log_steps)
                res_dirs = [run_path + '/results/' + res_dir for res_dir in res_dirs]
                num_steps = len(res_dirs)
                log_steps = []
                for res_dir in res_dirs:
                    with open(res_dir, 'rb') as file:
                        log_step = pickle.load(file)
                        log_steps.append(log_step)
                run = {}
                run['hparams'] = hyperparams
                run['res_steps'] = log_steps
                run['num_steps'] = num_steps
                #print(hyperparams)
                #print(num_steps)
                run_res.append(run)

    runs = run_res
    total_keys = []
    for run in runs:
        assert type(vars(run['hparams']))==dict
        total_keys = total_keys + list(vars(run['hparams']).keys())
        first_log_step = run['res_steps'][0]
        log_step_keys = list(first_log_step.keys())
        total_keys = total_keys + log_step_keys
        total_keys = list(set(total_keys)) # remove duplicate keys

    steps = []
    for run in runs:

        for i, res_step in enumerate(run['res_steps']):
            assert type(res_step)==dict
            sdict = {}
            for key in vars(run['hparams']).keys():
                val = vars(run['hparams'])[key]
                sdict['hparam_'+key] = val

            for key in res_step.keys():
                val = res_step[key]
                sdict[key] = val
            sdict['run_steps'] = run['num_steps']
            sdict['step'] = i
            steps.append(sdict)
    df = pd.DataFrame(steps)
    return df

def eval_data(df):
    df_keys = [key for key in df.keys() if ((not ('hparam_run_key' in key) or ('step' in key)))]
    df = pd.melt(df, id_vars=['hparam_run_key', 'run_steps', 'hparam_experiment', 'step'], value_vars=df_keys, var_name='measurement')

    def describe_sender(series):
        if 'sender' in series['measurement']:
            sender_idx = [int(sender) for sender in series['measurement'].split('_') if sender.isnumeric()][0]
        else:
            sender_idx = np.nan
        return sender_idx

    def describe_receiver(series):
        if 'receiver' in series['measurement']:
            receiver_idx = [int(receiver) for receiver in series['measurement'].split('_') if receiver.isnumeric()][1]
        else:
            receiver_idx = np.nan
        return receiver_idx

    def mode(series):
        if 'train' in series['measurement']:
            m='train'
        elif 'test' in series['measurement']:
            m='test'
        else:
            m=np.nan
        return m

    def measurement(series):
        if 'acc' in series['measurement']:
            m = 'acc'
        elif 'loss' in series['measurement']:
            m = 'loss'
        elif 'entropy' in series['measurement']:
            m = 'entropy'
        else:
            print(series['measurement'])
            m = np.nan
        return m


    df['sender'] = df.apply(describe_sender, axis=1)
    df['receiver'] = df.apply(describe_receiver, axis=1)
    df['eval_type'] = df.apply(mode, axis=1)
    df['measurement'] = df.apply(measurement, axis=1)


    return df






def extract_results(df, min_steps, pop_size):
    df = df[df.run_steps>=min_steps]
    df = df[df.hparam_num_receivers == pop_size]
    def is_relevant_col_name(name):
        #Different conditions on which we want to keep columns
        if ('sender' in name) and ('receiver' in name):
            return True
        if 'entropy' in name:
            return True
        if ('hparam_run_key' in name) or ('step' in name) or ('experiment' in name):
            return True
        return False
    rel_res = [col_name for col_name in list(df) if is_relevant_col_name(col_name)]
    df = df.filter(items=rel_res)

    return df


"""
def main(dirs):


    res_directories = ['results/' + dir for dir in dirs]
    runs = [[res_dir + '/' + run_dir for run_dir in os.listdir(res_dir)] for res_dir in res_directories]
    run_res = []
    for run_type in runs:
        for setting_path in run_type:
            for run_path in os.listdir(setting_path):
                print(run_path)
                run_path = setting_path + '/' + run_path
                with open(run_path + '/hyperparameters.pickle', 'rb') as file:
                    hyperparams = pickle.load(file)
                res_dirs = os.listdir(run_path + '/results')
                res_dirs = [res_dir for res_dir in res_dirs if res_dir.startswith('log_step')]
                res_dirs = sorted(res_dirs, key=sort_key_log_steps)
                res_dirs = [run_path + '/results/' + res_dir for res_dir in res_dirs]
                num_steps = len(res_dirs)
                log_steps = []
                for res_dir in res_dirs:
                    with open(res_dir, 'rb') as file:
                        log_step = pickle.load(file)
                        log_steps.append(log_step)
                run = {}
                run['hparams'] = hyperparams
                run['res_steps'] = log_steps
                run['run_steps'] = num_steps
                print(hyperparams)
                print(num_steps)
                run_res.append(run)

    data = analyze_baseline_vs_tscl_success(runs=run_res)
    return data
"""

def main():
    df = runs_dataframe()
    with open('analysis_data.pickle', 'wb') as file:
        pickle.dump(df, file)
    """
    
    pop_sizes = [2,5,10]
    res = []
    for pop_size in pop_sizes:
        dfp = extract_results(df, 100, pop_size)
        dfp = eval_data(dfp)
        res.append(dfp)
    return res[0], res[1], res[2]
    """
if __name__ == '__main__':
    main()
    """
    parser = argparse.ArgumentParser(description='Analysis argparse')
    parser.add_argument('dirs', nargs='+')
    args = parser.parse_args()
    main(args.dirs)
    """










