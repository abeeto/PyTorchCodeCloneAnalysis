import pytorch_lightning as pl
from finetuner import LoraBertFinetuner
from pytorch_lightning import Trainer
from argparse import ArgumentParser
from configparser import ConfigParser
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

def load_config(from_config=True,config_name='config.ini'):
    parser = ArgumentParser()
    # parser = Trainer.add_argparse_args(parser)
    parser.add_argument("--run_id", default="runs", help="runs name", type=str)
    parser.add_argument("--task", default="sst2", help="task", type=str)
    parser.add_argument("--model_id", default="bert-base-cased", help="task", type=str)
    parser.add_argument("--batch", default=32, help="batch size", type=int)
    parser.add_argument("--lr", default=2e-5, help="learning rate", type=float)
    parser.add_argument("--weight_decay", default=0, help="L2 regularization", type=float)
    parser.add_argument("--warmup_ratio", default=0.06, help="warmup learning rate", type=float)
    parser.add_argument("--warmup_steps", default=0, help="warmup learning rate", type=int)
    parser.add_argument("--max_updates", default=0, help="max update", type=int)
    parser.add_argument("--epoch", default=3, help="epoch", type=int)
    parser.add_argument("--gpus", default=1, help="gpus", type=int)
    parser.add_argument("--lora", default=0, help="lora", type=int)
    parser.add_argument("--alpha", default=0.1, help="alpha", type=float)
    parser.add_argument("--r", default=1, help="r", type=int)
    parser.add_argument("--log_dir", default=None, help="log path",type=str)
    parser.add_argument("--lora_path", default=None, help="lora path",type=str)
    parser.add_argument("--max_seq_length", default=128, help="lora path",type=int)
    parser.add_argument("--early_stop", default=0, help="lora path",type=int)
    parser.add_argument("--model_check", default=0, help="lora path",type=int)
    parser.add_argument("--test_only", default=0, help="lora path",type=int)

    # only when you can't use command line(overwrite cmd)
    if from_config:
        config = ConfigParser()
        config.read('./config/'+config_name)
        args_list = []
        for k,v in config['train'].items():
            args_list.append('--'+k)
            args_list.append(v)

        args = parser.parse_args(args_list)
    else:
        args = parser.parse_args()
    
    return args


def main(args):
    bert_finetuner = LoraBertFinetuner(args)

    tb_logger = TensorBoardLogger(args.log_dir, name=args.run_id)
    csv_logger = CSVLogger(args.log_dir, name=args.run_id)

    trainer = pl.Trainer(
        gpus=args.gpus,
        max_epochs=args.epoch,
        default_root_dir=args.log_dir,
        logger=[tb_logger, csv_logger],
    )
    if not args.test_only:
        trainer.fit(bert_finetuner)
        trainer.test(ckpt_path='best', verbose=True)
        if args.lora:
            bert_finetuner.save_lora_params(checkpoint_path="./logs/lora-ft/lora_params_last.ckpt")
    else: 
        trainer.test(ckpt_path='best', verbose=True)

if __name__ == "__main__":
    # True if run from python run_lora.py
    # Flase if run from bash run_lora.sh
    #args = load_config(True,'bert-config.ini')
    #args = load_config(True,'roberta-config.ini')
    args = load_config(True,'lora-config-1.ini')
    print(args)

    main(args)
    
#with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False, profile_memory=False) as prof:
#     bert_finetuner.train_dataloader()
#    outputs = bert_finetuner.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
#    print(prof.table())
