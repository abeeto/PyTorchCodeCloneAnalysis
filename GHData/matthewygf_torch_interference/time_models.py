from models_def import *
import csv
import subprocess
import time
import datetime
import os
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
    'pnasb_cmd': pnasb_cmd,
    'pos_cmd': pos_cmd,
    'mt1_cmd': mt1_cmd,
    'mt2_cmd': mt2_cmd,
    'lm_cmd': lm_cmd,
    'lm_large_cmd': lm_large_cmd,
    'lm_med_cmd': lm_med_cmd,
}
with open("timing_models_2080.csv", "w+") as f:
  header = ["model", "batch", "time"]
  csv_writer = csv.DictWriter(f, header, delimiter=',', lineterminator='\n')
  csv_writer.writeheader()

  for k,v in models_train.items():
    if "lstm" in v or "transformer" in v:
      bses = [16,32,64]
    else:
      bses = [64,128,256]

    curr_dir = os.path.abspath(os.path.dirname(__file__))
    execution_id = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
    data_dir = ['--dataset_dir', curr_dir]
    for b in bses:
      r_name = execution_id+k+str(b)
      cmd = ['--run_name', r_name, '--batch_size', str(b)] + data_dir
      cmd = v + cmd
      start_time = time.time()
      with open(r_name+".txt", "w+") as rf:
        p = subprocess.Popen(cmd, stdout=rf, stderr=rf)
        while p.poll() == None:
          time.sleep(1)
      elapsed = time.time() - start_time
      try:
        csv_writer.writerow({'model':k, 'batch': str(b),'application_runtime(s)': str(elapsed)})
      except:
        print("%s failed to write" % k)
        break

    
