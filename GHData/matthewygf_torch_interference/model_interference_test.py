
import os
import signal
import time
import subprocess
import datetime
import time
import logging
import system_tracker as sys_track
import numpy as np
import copy
import models_to_run
from models_def import *

# NOTE: CNNs
_default_batch_size = [64]

# NOTE: MISCs
nvprof_prefix_cmd = ['nvprof', '--profile-from-start', 'off', '--csv',]
pmon_mod_cmd = ['pmon', '--csv=true', '--interval=250', '--once=false']
pcie_mod_cmd = ['pcie', '--csv=true', '--interval=250']

models_train = {
    'mnasnet0_5_cmd': mnasnet0_5_cmd,
    'mnasnet1_0_cmd': mnasnet1_0_cmd,
    'mnasnet1_3_cmd': mnasnet1_3_cmd,
    'dpn92_cmd': dpn92_cmd,
    'dpn26_cmd': dpn26_cmd,
    'dpn26small_cmd': dpn26small_cmd,
    'pyramidnet_48_110_cmd': pyramidnet_48_110_cmd,
    'pyramidnet_84_110_cmd': pyramidnet_84_110_cmd,
    'pyramidnet_84_66_cmd': pyramidnet_84_66_cmd,
    'pyramidnet_270_110_bottleneck_cmd': pyramidnet_270_110_bottleneck_cmd,
    'resnet_wide_18_2_bottleneck_cmd': resnet_wide_18_2_bottleneck_cmd,
    'alexnet_cmd': alex_cmd,
    'googlenet_cmd': googlenet_cmd,
    'squeezenetv1_0_cmd': squeezenetv1_0_cmd,
    'inceptionv3_cmd': inceptionv3_cmd,
    'mobilenetv2_cmd': mobilenetv2_cmd,
    'mobilenetv2_large_cmd': mobilenetv2_large_cmd,
    'densenet121_cmd': dense121_cmd,
    'densenet161_cmd':dense161_cmd, 
    'densenet169_cmd': dense169_cmd,
    'vgg11_cmd': vgg11_cmd,
    'vgg11bn_cmd': vgg11bn_cmd,
    'vgg19_cmd': vgg19_cmd,
    'resnet18_cmd': resnet18_cmd,
    'resnet34_cmd': resnet34_cmd,
    'resnet50_cmd': resnet50_cmd,
    'resnext29_2x64_cmd': resnext29_2x64_cmd,
    'resnext11_2x64_cmd': resnext11_2x64_cmd,
    'resnext11_2x16_cmd': resnext11_2x16_cmd,
    'shufflenet_2_0_cmd': shufflenet_2_0_cmd,
    'shufflenet_1_0_cmd': shufflenet_1_0_cmd,
    'shufflenet_0_5_cmd': shufflenet_0_5_cmd,
    # 'efficientnetb4_cmd': eff_b4_cmd, 
    # 'efficientnetb3_cmd': eff_b3_cmd,
    # 'efficientnetb2_cmd': eff_b2_cmd,
    # 'efficientnetb1_cmd': eff_b1_cmd,
    # 'efficientnetb0_cmd': eff_b0_cmd,
    'pnasb_cmd': pnasb_cmd,
    'pos_cmd': pos_cmd,
    'mt1_cmd': mt1_cmd,
    'mt2_cmd': mt2_cmd,
    'lm_cmd': lm_cmd,
    'lm_large_cmd': lm_large_cmd,
    'lm_med_cmd': lm_med_cmd,
    'nvprof_prefix': nvprof_prefix_cmd,
    'pmon_mod_cmd': pmon_mod_cmd,
    'pcie_mod_cmd': pcie_mod_cmd
}

def process(line):
    # assuming always have sec/step
    if 'sec/step' in line:
        return line.split('(', 1)[1].split('sec')[0]
    else:
        return 0.0

def get_average_num_step(file_path):
    num = 0.0
    mean = 0.0
    with open(file_path, 'r') as f:
        for line in f:
            if 'sec/step' in line:
                mean = mean * num
                time_elapsed = process(line)
                num += 1
                mean = (mean + float(time_elapsed)) / num
    return (num, mean)

def create_process(batch_size, model_name, index, experiment_path, percent=0.0, is_nvprof=False, nvprof_args=None, gpu=None):
    execution_id = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
    output_dir_name = execution_id+model_name+str(index)
    if is_nvprof:
        output_dir_name = 'nvprof' + output_dir_name
    output_dir = os.path.join(experiment_path, output_dir_name)
    output_file = os.path.join(output_dir, 'output.log') 
    err_out_file = os.path.join(output_dir, 'err.log') 
    train_dir = os.path.join(output_dir, 'experiment')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    err = open(err_out_file, 'w+')
    out = open(output_file, 'w+')
    cmd = None
    cmd = copy.deepcopy(models_train[model_name])
    curr_dir = os.path.abspath(os.path.dirname(__file__))
    cmd = cmd + ['--dataset_dir', curr_dir]
    cmd = cmd + ['--run_name', output_dir_name]
    cmd = cmd + ['--batch_size', str(batch_size)]
    if gpu is not None:
      cmd = cmd + ['--device', str(gpu)]
    if _PROF_ONLY:
        cmd = cmd + ['--profile_only']

    if is_nvprof and not _PROF_ONLY:
        nvprof_log = os.path.join(train_dir, str(index)+model_name+'nvprof_log.log')
        nv_prefix = copy.deepcopy(models_train['nvprof_prefix'])
        nv_prefix += ['--log-file', nvprof_log]
        if nvprof_args is not None:
            nv_prefix += nvprof_args
        cmd = nv_prefix + cmd
    
    print(cmd)
    p = subprocess.Popen(cmd, stdout=out, stderr=err)
    return (p, out, err, err_out_file, output_dir)

def kill_process_safe(pid, 
                      err_handle, 
                      out_handle, 
                      path, 
                      ids, 
                      accumulated_models, 
                      mean_num_models,
                      mean_time_p_steps,
                      processes_list,
                      err_logs,
                      out_logs,
                      start_times,
                      err_file_paths,
                      i):
    err_handle.close()
    out_handle.close()
    path_i = path
    num, mean = get_average_num_step(path_i)
    model_index = ids[pid]
    mean_num_models[model_index] = ((accumulated_models[model_index] * mean_num_models[model_index]) + num) / (accumulated_models[model_index] + 1.0)
    mean_time_p_steps[model_index] = ((accumulated_models[model_index] * mean_time_p_steps[model_index]) + mean) / (accumulated_models[model_index] + 1.0)
    accumulated_models[model_index] += 1.0
    processes_list.pop(i)
    err_logs.pop(i)
    out_logs.pop(i)
    start_times.pop(i)
    err_file_paths.pop(i)
    return mean, num
    
_RUNS_PER_SET = 1
_START = 1
_RUN_NVPROF = False
_PROF_ONLY = False 
GPU = 1

def run(
    batch_size,
    average_log, experiment_path, 
    experiment_set, total_length, 
    experiment_index,
    nvprofiling=False):
    
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)

    mean_num_models = np.zeros(len(experiment_set), dtype=float)
    mean_time_p_steps = np.zeros(len(experiment_set), dtype=float)
    accumulated_models = np.zeros(len(experiment_set), dtype=float)

    is_single = len(experiment_set) == 1

    if is_single and nvprofiling:
        # 1. we want to use nvprof three times at least, make sure the metrics are correct
        for metric_run in range(3):
          nvp, out, err, path, out_dir = create_process(experiment_set[0], 1, experiment_path, percent=0.92, is_nvprof=True, 
              nvprof_args=['--timeout', str(60*7), '--metrics', 
                'achieved_occupancy,ipc,sm_efficiency,dram_utilization,sysmem_utilization,flop_dp_efficiency,flop_sp_efficiency',],
              gpu=GPU)
          while nvp.poll() is None:
              print("nvprof profiling metrics %s" % experiment_set[0])
              time.sleep(2)
          out.close()
          err.close()

    for experiment_run in range(_START, _RUNS_PER_SET+_START):
        if not _PROF_ONLY:
            if os.path.exists(average_log):
                average_file = open(average_log, mode='a+')
            else:
                average_file = open(average_log, mode='w+')
        processes_list = []
        err_logs = []
        out_logs = []
        out_file_paths = []
        start_times = []
        ids = {}
        pmon_log_path = os.path.join(experiment_path, str(experiment_run)+'pmon.log')
        pmon_log = open(pmon_log_path, 'a+')
        pmon_csv = os.path.join(experiment_path, str(experiment_run)+'pmon.csv')
        pmon_cmd = copy.deepcopy(models_train['pmon_mod_cmd'])
        pmon_cmd += ['--logpath='+pmon_csv]
        pmon_p = subprocess.Popen(pmon_cmd, stdout=pmon_log, stderr=pmon_log)
        pmon_poll = None
        percent = (1 / len(experiment_set)) - 0.075 # some overhead of cuda stuff i think :/
        for i, m in enumerate(experiment_set):
            if i > 0:
              time.sleep(20)
            start_time = time.time()
            p, out, err, path, out_dir = create_process(batch_size, m, i, experiment_path, percent=percent, gpu=GPU)
            processes_list.append(p)
            err_logs.append(err)
            out_logs.append(out)
            start_times.append(start_time)
            out_file_paths.append(path)
            ids[p.pid] = i
        should_stop = False
        if not _PROF_ONLY:
            sys_tracker = sys_track.InfosTracker(experiment_path)

        try:
            if not _PROF_ONLY:
                

                pcie_log_path = os.path.join(experiment_path, str(experiment_run)+'pcie.log')
                pcie_log = open(pcie_log_path, 'a+')
                pcie_csv = os.path.join(experiment_path, str(experiment_run)+'pcie.csv')
                pcie_cmd = copy.deepcopy(models_train['pcie_mod_cmd'])
                pcie_cmd += [ '--logpath='+pcie_csv ]
                pcie_p = subprocess.Popen(pcie_cmd, stdout=pcie_log, stderr=pcie_log)
                pcie_poll = None

                smi_file_path = os.path.join(experiment_path, str(experiment_run)+'smi_out.log') 
                smi_file = open(smi_file_path, 'a+')
                nvidia_csv = "smi_watch.csv"
                nvidia_csv = str(experiment_run)+nvidia_csv
                nvidia_smi_cmd = ['watch', '-n', '0.2', 'nvidia-smi', 
                                '--query-gpu=memory.used,memory.total,utilization.gpu,utilization.memory,power.draw', 
                                '--format=noheader,csv', '|', 'tee', '-a' , experiment_path+'/'+nvidia_csv]
                smi_p = subprocess.Popen(nvidia_smi_cmd, stdout=smi_file, stderr=smi_file)
                smi_poll = None
                sys_tracker.start()
            while not should_stop:
                time.sleep(5)
                if len(processes_list) <= 0:
                    should_stop = True

                for i,(p, err, out, start_time, path) in enumerate(zip(processes_list, err_logs, out_logs, start_times, out_file_paths)):
                    poll = None
                    pid = p.pid
                    poll = p.poll()
                    current_time = time.time()
                    executed = current_time - start_time
                    if poll is None:
                        print('Process %d still running' % pid)
                    else:
                        mean, num = kill_process_safe(pid, err, out, path, ids, accumulated_models, 
                                                    mean_num_models, mean_time_p_steps, processes_list, err_logs, out_logs, start_times, out_file_paths, i)
                        if not _PROF_ONLY:
                            line = ("experiment set %d, experiment_run %d: %d process average num p step is %.4f and total number of step is: %d \n" % 
                                    (experiment_index, experiment_run, pid, mean, num))
                            average_file.write(line)

                if not _PROF_ONLY:
                    smi_poll = smi_p.poll()
                    if smi_poll is None:
                        print('NVIDIA_SMI Process %d still running' % smi_p.pid)
                    
                    pcie_poll = pcie_p.poll()
                    if pcie_poll is None:
                        print('PCIe mon Process %d still running' % pcie_p.pid)


                pmon_poll = pmon_p.poll()
                if pmon_poll is None:
                    print('PMON Process %d still running' % pmon_p.pid)

            print('total experiments: %d, experiment_run %d , finished %d' % (total_length-1, experiment_run, experiment_index))

        except KeyboardInterrupt:
            if not _PROF_ONLY:
                smi_p.kill()
                smi_file.close()
                for p, err, out in zip(processes_list, err_logs, out_logs):
                    pid = p.pid
                    p.kill()
                    err.close()
                    out.close()
                    print('%d killed ! ! !' % pid)
            else:
                print("done")
        finally:
            print("final")
            pmon_p.kill()
            pmon_log.close()
            if not _PROF_ONLY:
                smi_poll = smi_p.poll()
                pcie_p.kill()
                pcie_log.close()
                if smi_poll is None:
                    smi_p.kill()
                    smi_file.close()
        if not _PROF_ONLY:
            average_file.close()
            sys_tracker.stop()
    if not _PROF_ONLY:
        # Experiment average size.
        average_file = open(average_log, mode='a+')
        for i in range(len(experiment_set)):
            average_file.write("TOTAL: In experiment %d average mean sec/step and average number for model %d are %.4f , %d \n" % 
                            (experiment_index, i, mean_time_p_steps[i], mean_num_models[i]))
        average_file.close()
    
def main():
    # which one we should run in parallel
    # TODO: randomly start each process.
    sets = copy.deepcopy(models_to_run.sets)
    curr_dir = None
    if os.name == "nt":
      curr_dir = os.getcwd()
    else:
      curr_dir = os.path.dirname(__file__)
    project_dir = os.path.abspath(os.path.dirname(curr_dir))
    for b in _default_batch_size:
        experiment_path = os.path.join(project_dir, 'experiment')
        experiment_path = experiment_path+str(b)
        for experiment_index, ex in enumerate(sets):
            current_experiment_path = os.path.join(experiment_path, str(experiment_index))
            experiment_file = os.path.join(experiment_path, 'experiment.log')

            if _RUN_NVPROF:
                # TODO: transition and test with nsight
                # NOTE: timeline please use nsight-system gui to do so. much better.
                current_experiment_path = os.path.join(current_experiment_path, "timeline_metrics")
                profiled_log = os.path.join(current_experiment_path, 'experiment.log')
                run(b, profiled_log, current_experiment_path, ex, len(sets), experiment_index, _RUN_NVPROF)
            else:
                run(b, experiment_file, current_experiment_path, ex, len(sets), experiment_index)

        if not _PROF_ONLY:
            app_csv_p = subprocess.Popen(['python', 'multiprocess_appfinishtime.py'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            while app_csv_p.poll() is None:
                time.sleep(2)
                print("waiting for app time csv finish")
            print("Done.")
if __name__ == "__main__":
    main()
        
