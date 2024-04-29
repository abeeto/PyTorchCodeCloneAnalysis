import yaml 
import glob 
import argparse
from subprocess import run

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, required=True,
                    help="choose one of: pets, flowers_102, birds, sculptures")
args = parser.parse_args()

tree = glob.glob('*')

if 'external_data' not in tree:
    run(f'mkdir external_dataset', shell=True)

config = yaml.load(open('configs/config.yaml'), Loader=yaml.FullLoader)

dataset = config['dataset'][args.dataset]

run(f'wget {dataset} -P external_dataset/', shell=True)