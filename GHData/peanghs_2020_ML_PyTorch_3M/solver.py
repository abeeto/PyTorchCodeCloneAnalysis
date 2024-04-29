import torch
import matplotlib.pyplot as plt
from model import chapter3_1, chapter3_2, chapter3_3, chapter4_1, chapter5_1

class Starter(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'현재 모드 : {self.device}')

    def load_model(self, args):
        mode = args.mode
        print(f'----{mode}를 시작합니다----')
        if mode == 'chapter_3_1':
            play_chapter_3 = chapter3_1.ClassChapter3_1()
            print(play_chapter_3.tensor_config())
            print(play_chapter_3.tensor_multiple())
            print(play_chapter_3.autograd_train())
        if mode == 'chapter_3_2':
            play_chapter_3 = chapter3_2.ClassChapter3_2()
            random_tensor = play_chapter_3.distance_calc(args)
            random_tensor = random_tensor.cpu()
            plt.imshow(random_tensor.view(100, 100).data)
            plt.show()
        if mode == 'chapter_3_3':
            play_chapter_3 = chapter3_3.ClassChapter3_3_eval()
            play_chapter_3.process(args)
        if mode == 'chapter_4_1':
            play_chapter_4 = chapter4_1.ClassChapter4_1()
            play_chapter_4.starter(args)
        if mode == 'chapter_5_1':
            play_chapter_5 = chapter5_1.ClassChapter5_1()
            play_chapter_5.starter(args)