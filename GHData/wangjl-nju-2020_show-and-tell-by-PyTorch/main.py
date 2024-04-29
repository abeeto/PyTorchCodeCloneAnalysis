from config import DATA_ROOT
from config import hparams
from process_cap import process_caption
from test import Tester
from train import Trainer
from vocab import dump_vocab


def main():
    dump_vocab(hparams.vocab_pkl, hparams.train_cap)
    process_caption(cap_dir=(DATA_ROOT + 'karpathy_split/'),
                    vocab_pkl=hparams.vocab_pkl,
                    max_sen_len=hparams.max_sen_len)
    trainer = Trainer(hparams)
    # trainer.validating(0, hparams.val_path)
    trainer.training(max_epochs=40,
                     val_interval=20,
                     save_path='/home/wangjl/work_dir/nic/best_model_1204.ckpt',
                     val_path='/home/wangjl/work_dir/nic/val_1204.json')
    trainer.fine_tune_encoder(fine_tune_epochs=10,
                              val_interval=5,
                              save_path='/home/wangjl/work_dir/nic/best_model_1204.ckpt',
                              val_path='/home/wangjl/work_dir/nic/val_1204.json')
    tester = Tester(hparams)
    tester.testing(save_path='/home/wangjl/work_dir/nic/best_model_1204.ckpt',
                   test_path='/home/wangjl/work_dir/nic/test_1204.json')


if __name__ == '__main__':
    main()
