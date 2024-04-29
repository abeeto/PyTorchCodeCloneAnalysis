import argparse
import os
from datasets.preprocessor import get_files, preprocess_data


def main():
    parser = argparse.ArgumentParser(description='WaveRNN-PyTorch Training')
    parser.add_argument('--data_dir', metavar='DIR', default='/home/lynn/dataset/LJSpeech-1.1/wavs',
                        help='path to dataset')
    parser.add_argument('--base_dir', metavar='DIR', default='/home/lynn/workspace/wumenglin/WaveRNN_pytorch',
                        help='path to dataset')
    parser.add_argument('--output_dir', metavar='DIR', default='dataset/',
                        help='path to dataset')
    parser.add_argument('-b', '--bits', default=9, type=int,
                        metavar='N', help='quantilization bits (default: 9)')

    args = parser.parse_args()
    output_path = os.path.join(args.base_dir, args.output_dir)
    mel_path = os.path.join(output_path, 'mels')
    quant_path = os.path.join(output_path, 'quant')
    os.makedirs(mel_path, exist_ok=True)
    os.makedirs(quant_path, exist_ok=True)

    wav_files = get_files(args.data_dir)
    preprocess_data(wav_files, output_path=output_path, mel_path=mel_path, quant_path=quant_path,
                                 bits=args.bits)


if __name__ == '__main__':
    main()