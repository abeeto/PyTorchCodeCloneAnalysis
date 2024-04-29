import argparse
import cv2
import numpy as np
import torch
import re
import os.path
import sys
import colour_demosaicing
import yaml
from glob import glob
from importlib import import_module
import matplotlib.pyplot as plt


class ConfigurationStruct:
    def __init__(self, entries: dict):
        self.__dict__.update(entries)
        for key, val in self.__dict__.items():
            if 'kargs' in key and val is None:
                self.__dict__[key] = {}


def get_class(class_name, file_path: str = None, module_path: str = None):
    if file_path:
        try:
            module = import_module(os.path.basename(file_path.replace('.py', '')))
        except ImportError:
            sys.path.append(os.path.dirname(file_path))
            module = import_module(os.path.basename(file_path.replace('.py', '')))
    elif module_path:
        module = import_module(module_path)
    else:
        raise Exception('module path or file path are required')
    return getattr(module, class_name)


def read_yaml(file_path):
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    with open(file_path) as f:
        data = yaml.load(f, Loader=loader)
    return data


def print_progress(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def pad_2d(img: np.ndarray, divisor) -> np.ndarray:
    """return padded image to shape of nearest divisor"""
    if np.sum(np.array(img.shape[:2]) % divisor) == 0:
        return img
    expand = divisor - (np.array(img.shape[:2]) % divisor)
    expand = np.bitwise_xor(expand, divisor)
    top, left = expand // 2
    bottom, right = expand - (expand // 2)
    if img.ndim > 2:
        return np.pad(img, pad_width=((top, bottom), (left, right), (0, 0)), mode='constant')
    else:
        return np.pad(img, pad_width=((top, bottom), (left, right)), mode='constant')


def crop_center(img, cropy, cropx):
    y, x = img.shape[:2]
    sx = x//2-(cropx//2) + cropx % 2
    sy = y//2-(cropy//2) + cropy % 2
    return img[sy:sy+cropy, sx:sx+cropx]


def inference_image(model, device, factors: np.ndarray, im_path: str, demosaic: bool, rotate: bool,
                    bit_depth: int, raw_result: bool, do_crop: bool, gray: bool, fliplr: bool):
    max_bit = (2**bit_depth) - 1
    if demosaic:
        im_raw = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
        # im_raw = cv2.demosaicing(im_raw, cv2.COLOR_BayerBG2RGB).astype(np.float32)
        im_raw = colour_demosaicing.demosaicing_CFA_Bayer_bilinear(im_raw, pattern='GRBG')
        # im_raw = cv2.demosaicing(im_raw, cv2.COLOR_BayerRG2RGB).astype(np.float32)

    elif gray:
        im_raw = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
    else:
        im_raw = cv2.imread(im_path)
        im_raw = cv2.cvtColor(im_raw, cv2.COLOR_BGR2RGB).astype(np.float32)

    if fliplr:
        im_raw = np.fliplr(im_raw)

    if factors is not None:
        im_raw = im_raw * factors

    in_img = np.clip(im_raw, 0, max_bit) / max_bit
    if rotate:
        in_img = np.rot90(in_img)
    org_shape = in_img.shape[:2]
    if do_crop:
        in_img = crop_center(in_img, min(2048, in_img.shape[0]), min(2048, in_img.shape[1]))
    return_img = (in_img * 255).astype(np.uint8)
    if not do_crop:
        in_img = pad_2d(in_img, 32).astype(np.float32)

    if in_img.ndim > 2:
        in_img = np.transpose(in_img, (2, 0, 1))
    inputs = torch.from_numpy(in_img).float().to(device)
    inputs = inputs.unsqueeze(0)
    if inputs.ndim < 4:
        inputs = inputs.unsqueeze(0)
    out = model(inputs)
    if raw_result:
        out_im = out.squeeze().cpu().numpy()
        out_im = out_im.transpose(1, 2, 0)
        return out_im, return_img
    if out.ndim > 3 and out.shape[1] > 3:  # segmentation
        outs = out.argmax(dim=1).squeeze()
        out_im = outs.cpu().numpy()
        out_im = out_im.astype(np.uint8)

    elif out.shape[1] == 3:  # im2im
        out_np = out.cpu().numpy()
        if out_np.ndim > 2:
            out_np = np.squeeze(out_np, axis=0)
            out_np = out_np.transpose((1, 2, 0))
        out_im = np.clip(out_np, 0, 1)
        out_im = (out_im * 255).astype(np.uint8)
    else:
        out_np = out.squeeze().cpu().numpy()
        out_im = np.clip(out_np, -4, 10)
        out_im = ((out_im + 4) * (255/15)).astype(np.uint8)

    # out_im = np.pad(out_im, pad_width=((2, 2), (1, 1), (0, 0)), mode='constant')
    # remove the padding
    if not do_crop:
        if out_im.shape[:2] != org_shape[:2]:
            out_im = crop_center(out_im, org_shape[0], org_shape[1])
    if fliplr:
        out_im = np.fliplr(out_im)
        return_img = np.fliplr(return_img)
    return out_im, return_img


def save_image(img: np.ndarray, in_im_path: str, model_name: str, out_dir: str):
    out_name = os.path.basename(in_im_path)
    out_name = out_name.split('.')[0] + '_' + model_name + '.png'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path = os.path.join(out_dir, out_name)
    out_im = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, out_im)


def display_result(in_img, out_img, rot90):
    if rot90:
        in_img = np.rot90(in_img)
        out_img = np.rot90(out_img)
    if out_img.ndim == 2:
        cmap = 'jet'
    else:
        cmap = 'gray'
    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
    axes[0].imshow(in_img), axes[0].set_title('in')
    axes[1].imshow(out_img, cmap=cmap), axes[1].set_title('out')
    plt.show()


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch training module',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model_path', help='path to pre-trained model')
    parser.add_argument('-i', '--images_path', nargs='+', help='list of image or folder of images')
    parser.add_argument('-w', '--out_path', default=None, help='path to output file')
    parser.add_argument('-g', '--gpu_index', help='index of gpu (if exist, torch indexing)', type=int, default=0)
    parser.add_argument('-dm', '--demosaic', action='store_true', default=False)
    parser.add_argument('-gr', '--gray', action='store_true', default=False)
    parser.add_argument('-t', '--rot90', action='store_true', default=False)
    parser.add_argument('-cr', '--do_crop', action='store_true', default=False)
    parser.add_argument('-lr', '--fliplr', action='store_true', default=False,
                        help='flip before inference and flip back afterwards - in order to use RGGB pattern on GRBG')
    parser.add_argument('-b', '--bit_depth', type=int, default=8, help='input image bit depth')
    parser.add_argument('-m', '--im_pattern', default='*wrapped_crop_wb.png', help='images regex pattern')
    # parser.add_argument('--check_patt', default='*mask.tif', help='input image file pattern')
    args = parser.parse_args()
    assert not (args.mat_out and not (args.mat_out ^ (args.out_type is None))), 'out path is required for mat'
    for in_path in args.images_path:
        assert os.path.exists(in_path), 'ERROR - path is not exist %s' % in_path

    return args


def load_model(model_path, device):
    # read configuration:
    config_dict = read_yaml(os.path.join(model_path, 'cfg.yaml'))
    for cfg_section, cfg_dict in config_dict.items():
        config_dict[cfg_section] = ConfigurationStruct(cfg_dict)
    config = ConfigurationStruct(config_dict)

    # create classes:
    model_cls = get_class(config.model.type, file_path=config.model.path)
    model = model_cls(**config.model.kargs)

    # read model
    cp_path = os.path.join(model_path, 'checkpoints', 'checkpoint.pth')
    print('loading checkpoint', cp_path)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.to(device)
    return model


def main():
    args = get_args()
    if int(args.gpu_index) >= 0 and torch.cuda.is_available():
        device = torch.device("cuda:" + str(args.gpu_index))
        print('using device: ', torch.cuda.get_device_name(device))
    else:
        device = torch.device("cpu")
        print('using cpu')
    model = load_model(args.model_path, device)
    model.eval()
    with torch.no_grad():
        if args.images_path:
            model_name = os.path.basename(os.path.normpath(args.model_path))
            images = []
            for in_path in args.images_path:
                if os.path.isdir(in_path):
                    # given path is a directory
                    glob_patt = os.path.join(in_path, args.im_pattern)
                    images.extend(glob(glob_patt))
                else:
                    images.append(in_path)
            if len(images) == 0:
                print("ERROR - images list is empty, check glob input: {}".format(glob_patt))
                sys.exit(1)
            for i, im_path in enumerate(images):
                print_progress(i, total=len(images), suffix='inference {}{}'.format(im_path, ' '*20), length=20)
                out_im, in_img = inference_image(model, device, im_path=im_path, demosaic=args.demosaic,
                                                 rotate=args.rot90, bit_depth=args.bit_depth, do_crop=args.do_crop,
                                                 gray=args.gray, fliplr=args.fliplr)
                if args.out_path is not None:
                    save_image(out_im, im_path, model_name, args.out_path)
                else:
                    display_result(in_img=in_img, out_im=out_im, rot90=args.rot90)
            print_progress(len(images), total=len(images), suffix='inferenced {} images {}'.format(len(images), ' ' * 80), length=20)

if __name__ == '__main__':
    main()
