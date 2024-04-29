import yaml 
from models.generator import Generator 
from models.discriminator import Discriminator
from utils.loss import Loss
from utils.dataloader import DataSampler
import glob 
from subprocess import run
from torch import optim
from utils.train import trainer 

tree = glob.glob('*')

if 'logs' not in tree:
    run(f'mkdir logs', shell=True)
    run(f'mkdir logs/chkpt', shell=True)
    run(f'mkdir logs/tensorboard', shell=True)

config = yaml.load(open('configs/config.yaml'), Loader=yaml.FullLoader)

checkpoint_dir = 'logs/' + config['log']['chkpt_dir']
logging_dir = 'logs/' + config['log']['log_dir']
num_samples = config['log']['num_samples']
log_interval = config['log']['log_interval']
save_freq = config['log']['save_freq']
save_everything = config['log']['save_everything']
path =config['train']['folder']

device = config['train']['device']
batch_size = config['train']['batch_size']
num_workers = config['train']['num_workers']
resolution = config['train']['resolution']
n_epochs = config['train']['epochs']

lr = config['optimizer']['lr']
beta1 = config['optimizer']['beta_1']
beta2 = config['optimizer']['beta_2']
amsgrad = config['optimizer']['amsgrad']




trainloader = DataSampler.build(path, batch_size, num_workers, resolution)

generator = Generator(resolution).to(device)
optim_g = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))


discriminator = Discriminator(resolution).to(device)
optim_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))



trainer(generator, discriminator, optim_g, optim_d, trainloader, n_epochs, device, log_interval, logging_dir, save_freq, checkpoint_dir, resolution, num_samples, save_everything)
