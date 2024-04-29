#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# License: © 2021 Achille-Tâm GUILCHARD All Rights Reserved
# Author: Achille-Tâm GUILCHARD
# Usage: python3 build_docker_and_launch_inference.py --workdir <DIR> --imgdir <DIR>

import os
import subprocess
import argparse
import shutil
from termcolor import colored

def parse_arguments():                                                                                                                                                                                                                                                
    parser = argparse.ArgumentParser(description='Automatic Launching of Inference (Detection and Classification)')
    parser.add_argument('--workdir', type=str, default="./", help='Where the entries of the program are stored.') 
    parser.add_argument('--imgdir',  type=str, default="./", help='Where the input images are stored.')                                                                                                                                                                                                                     
    return parser.parse_args()  

print("")
print(colored('//////////////////////', 'blue'),colored('///////////////////////', 'white'),colored('//////////////////////', 'red'))
print(colored('//', 'blue'),  colored('Automatic Launching of Inference (Detection and Classification)', 'white'), colored('//', 'red'))
print(colored('//////////////////////', 'blue'),colored('///////////////////////', 'white'),colored('//////////////////////', 'red'))
print(colored('© 2021 Achille-Tâm GUILCHARD All Rights Reserved', 'red'))
print("")

args = parse_arguments()
workDir        = args.workdir
imgDir         = args.imgdir
nb_cpu         = 7
docker_run_cmd = 'docker run --rm -it --cpuset-cpus="0-' + str(nb_cpu) +'" -u $(id -u) -v ' + workDir + ':/tmp inference_detection_classification:latest '

print(colored('Entries summary', 'green'))
print("  > workDir: " + str(workDir))
print("  > imgDir:  " + str(imgDir))

imgDir = os.path.basename(imgDir)

# building of the docker
print("")
print(colored('/////////', 'blue'),colored('////////', 'white'),colored('/////////', 'red'))
print(colored('//', 'blue'),  colored('Building of the docker', 'white'), colored('//', 'red'))
print(colored('/////////', 'blue'),colored('////////', 'white'),colored('/////////', 'red'))
print("")
cmd_build_docker = "docker build -t inference_detection_classification:latest ."
error_code = subprocess.call(cmd_build_docker, shell=True)
if error_code != 0:
    print('Building of docker failed!')
    exit(error_code)

# Launch inference
print("")
print(colored('///////', 'blue'),colored('///////', 'white'),colored('////////', 'red'))
print(colored('//', 'blue'),  colored('Lauching training!', 'white'), colored('//', 'red'))
print(colored('///////', 'blue'),colored('///////', 'white'),colored('////////', 'red'))
print("")

# Create output directory
shutil.rmtree(workDir + "/results", ignore_errors=True)
os.makedirs(workDir   + "/results", exist_ok=True)

# Launch inference
cmd_launch_inference = docker_run_cmd + 'python3 /tmp/inference_detection_classification.py --input /tmp/' + imgDir + ' --output /tmp/results'
print(colored("launching " + cmd_launch_inference, 'red'))
print("")
if subprocess.call(cmd_launch_inference, shell=True) != 0:
    print('Inference failed!')
    exit(-1)
print("")