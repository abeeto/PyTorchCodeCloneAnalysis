import os
import argparse
from collections import OrderedDict
import numpy as np
import pandas as pd
import jittor as jt
from cleanfid import fid
from PIL import Image
from training.networks.stylegan2 import Generator

def save_image_pyjt(img, save_dir, name, Gray=False):
    fname, fext = name.split('.')
    imgPath = os.path.join(save_dir, "%s.%s" % (fname, fext))
    img_array = tensor2array(img.data[0])
    image_pil = Image.fromarray(img_array)
    if Gray:
        image_pil.convert('L').save(imgPath)  # Gray = 0.29900 * R + 0.58700 * G + 0.11400 * B
    else:
        image_pil.save(imgPath)


def tensor2array(image_tensor, imtype=np.float32, normalize=True):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2array(image_tensor[i], imtype, normalize))
        return image_numpy
    image_numpy = image_tensor.cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:
        image_numpy = image_numpy[:, :, 0]
    return image_numpy.astype(imtype)

def make_eval_images(g, save_folder, eval_samples, batch_size):
    if not os.path.exists(f'{save_folder}/image/'):
        os.makedirs(f'{save_folder}/image/')

    iterations = int(np.ceil(eval_samples / batch_size))
    images_left = eval_samples
    img_count = 0
    for i in range(iterations):
        batch = min(batch_size, images_left)
        images_left -= batch_size
        noise = jt.randn(batch, 512)
        sample, _ = g([noise])

        for ind in range(sample.size(0)):
            save_image_pyjt(sample[ind], f'{save_folder}/image/', f'{str(img_count).zfill(6)}.png')
            img_count += 1


def get_metrics(opt, name, target):
    real_folder = f"{opt.eval_root}/{target}/"
    fake_folder = f"{opt.sample_root}/{name}/"
    ckpt_path = f"{opt.model_root}/{name}.pth"
    g = setup_generator(ckpt_path)
    fid_value = fid.compute_fid(real_folder+'image', fake_folder+'image', num_workers=0)
    del g
    '''
    fake_feats, real_feats = stats_fake['vgg_features'], stats_real['vgg_features']
    with mp.Pool(1) as p:
        precision, recall = p.apply(run_precision_recall, (real_feats, fake_feats))
    '''
    return {
        "fid": fid_value,
        #"precision": precision,
        #"recall": recall
    }


def setup_generator(ckpt_path, w_shift=False):
    g = Generator(256, 512, 8, w_shift=w_shift)
    ckpt = jt.load(ckpt_path)
    g.load_state_dict(ckpt)
    g.eval()
    return g


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_list', type=str)
    parser.add_argument('--output', type=str, default='metric_results.csv')
    parser.add_argument('--model_root', type=str, default='./weights/')
    parser.add_argument('--eval_root', type=str, default='./data/eval/')
    parser.add_argument('--sample_root', type=str, default='./cache_files/')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--eval_samples', type=int, default=2500)
    parser.add_argument('--device', type=str, default='cuda')
    opt = parser.parse_args()

    with open(opt.models_list, 'r') as f:
        lst = [s.strip().split(' ') for s in f.readlines()]
        all_models, all_targets = zip(*lst)

    with jt.no_grad():
        metrics = OrderedDict()
        for name, target in zip(all_models, all_targets):
            metrics[name] = get_metrics(opt, name, target)
            print(f"({name}) {metrics[name]}")

            table_columns = ['fid']
            table = pd.DataFrame.from_dict(metrics, orient='index', columns=table_columns)
            table.to_csv(opt.output, na_rep='--')
