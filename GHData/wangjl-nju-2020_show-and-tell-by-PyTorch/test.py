import torch
import json

from data_loaders import get_test_loader
from models import CNN, RNN
from metrics import coco_eval
from tqdm import tqdm
from config import hparams

DEVICE = ('cuda: 0' if torch.cuda.is_available() else 'cpu')


class Tester:
    """
    测试
    """

    def __init__(self, _hparams):
        self.test_loader = get_test_loader(_hparams)
        self.encoder = CNN().to(DEVICE)
        self.decoder = RNN(fea_dim=_hparams.fea_dim,
                           embed_dim=_hparams.embed_dim,
                           hid_dim=_hparams.hid_dim,
                           max_sen_len=_hparams.max_sen_len,
                           vocab_pkl=_hparams.vocab_pkl).to(DEVICE)
        self.test_cap = _hparams.test_cap

    def testing(self, save_path, test_path):
        """
        测试

        :param save_path: 模型的保存地址
        :param test_path: 保存测试过程生成句子的路径
        :return:
        """
        print('*' * 20, 'test', '*' * 20)
        self.load_models(save_path)
        self.set_eval()

        sen_json = []
        with torch.no_grad():
            for val_step, (img, img_id) in tqdm(enumerate(self.test_loader)):
                img = img.to(DEVICE)
                features = self.encoder.forward(img)
                sens, _ = self.decoder.sample(features)
                sen_json.append({'image_id': int(img_id), 'caption': sens[0]})

        with open(test_path, 'w') as f:
            json.dump(sen_json, f)

        result = coco_eval(self.test_cap, test_path)
        for metric, score in result:
            print(metric, score)

    def load_models(self, save_path):
        ckpt = torch.load(save_path, map_location={'cuda:2': 'cuda:0'})  # 映射是因为解决保存模型的卡与加载模型的卡不一致的问题
        encoder_state_dict = ckpt['encoder_state_dict']
        self.encoder.load_state_dict(encoder_state_dict)
        decoder_state_dict = ckpt['decoder_state_dict']
        self.decoder.load_state_dict(decoder_state_dict)

    def set_eval(self):
        self.encoder.eval()
        self.decoder.eval()


if __name__ == '__main__':
    tester = Tester(hparams)
    tester.testing(save_path='/home/wangjl/work_dir/nic/best_model_1204.ckpt',
                   test_path='/home/wangjl/work_dir/nic/test_1204.json')
