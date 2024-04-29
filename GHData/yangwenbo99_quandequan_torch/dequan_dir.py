from PIL import Image
import argparse
from pathlib import Path
from quandequan import functional as QF
import numpy as np

ALLOWED_FORMAT = ['.png', '.bmp', '.jpg', 'spi']

def parse_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', type=str, help='directory of input')
    parser.add_argument('-o', '--output', type=str,
            help='directory of output, must DNE or be empty')

    config = parser.parse_args()
    return config

def travel(input_path: Path, output_path: Path, filecallback=None, _root_path=None):
    '''
    Warning: the file structure must be acyclic, otherwise there will be an
    endless recursion.

    @param filecallback: a function receiving (input_path, output_path)
                         for each file
    '''
    if not _root_path:
        _root_path = input_path
    if not output_path.is_dir():
        output_path.mkdir(parents=True)
    if len([x for x in output_path.iterdir()]) != 0:
        raise ValueError('The output directory must be empty.')
    for item in input_path.iterdir():
        diff = item.relative_to(input_path)
        print(diff)
        output_sub_path = output_path / diff
        if item.is_dir():
            travel(item, output_sub_path, filecallback=filecallback, _root_path=_root_path)
        elif item.is_file() and item.suffix.lower() in ALLOWED_FORMAT:
            if filecallback: filecallback(item, output_sub_path)

def convert(input_path: Path, output_path: Path):
    print(input_path, '->', output_path)
    img = Image.open(input_path)
    if img.mode == 'RGB':
        npimg = np.array(img)
        npimg = QF.random_dequantize(npimg)
    elif img.mode == 'F':
        npimg = numpy.array(img).trans
    else:
        raise ValueError('Unsupported image mode')
    npimg = np.transpose(npimg, (2, 0, 1))
    output_path = output_path.with_suffix('.tiff')
    QF.save_img(npimg, output_path)


def main(config):
    input_path = Path(config.input)
    output_path = Path(config.output)
    travel(input_path, output_path, convert)

if __name__ == "__main__":
    config = parse_config()
    main(config)

