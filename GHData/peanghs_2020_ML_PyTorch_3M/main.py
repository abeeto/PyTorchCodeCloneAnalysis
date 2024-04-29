import argparse
from solver import Starter


def main(args):
    starter = Starter(args)
    if args.mode == 'chapter_9_1':  # 다른 조건이 필요할 때 사용
        pass
    else:
        starter.load_model(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # ---- 설정 ----
    parser.add_argument('--mode', type=str, default='chapter_5_1',
                        choices=['chapter_3_1', 'chapter_3_2', 'chapter_3_3','chapter_4_1', 'chapter_5_1'])
    parser.add_argument('--lr', type=float, default='0.01', help='learning rate')
    parser.add_argument('--epochs', type=int, default='100')
    parser.add_argument('--batch_size', type=int, default='64')
    parser.add_argument('--drop_out', type=float, default='0.1')
    parser.add_argument('--momentum', type=float, default='0.5', help='SGD의 관성')

    args = parser.parse_args()
    print(args)
    main(args)
