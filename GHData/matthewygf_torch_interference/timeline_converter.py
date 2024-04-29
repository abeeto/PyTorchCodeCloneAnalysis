import os
import subprocess
import time

_TIMELINENAME = "_timeline"
_SUFFIX = "_conv.csv"

curr_dir = os.path.abspath(os.path.dirname(__file__))

is_windows = os.name == 'nt'
# assume your path is correct.
nvprof_bin = 'nvprof.exe' if is_windows else 'nvprof'
nvprof_convert = [nvprof_bin, '--print-gpu-trace', '--csv']
for dirpath, dirs, files in  os.walk(curr_dir):
    for filename in files:
        if _TIMELINENAME in filename:
            file_path = os.path.join(dirpath, filename)
            file_dir = os.path.dirname(file_path)
            new_file_name = os.path.join(file_dir, filename+_SUFFIX)
            nvprof_convert += ['-i', file_path, '--log-file', new_file_name]
            file_log = os.path.join(file_dir, 'conv.log')
            with open(file_log, 'a+') as file_log_handle:
              conv_p = subprocess.Popen(nvprof_convert,stderr=file_log_handle, stdout=file_log_handle)
              while conv_p.poll() is None:
                  print('still converting')
                  time.sleep(5)
              print('done %s' % filename)