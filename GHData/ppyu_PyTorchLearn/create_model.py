# -*- coding: utf-8 -*-
"""
@File   : create_model.py
@Author : Pengy
@Date   : 2020/10/13
@Description : Input your description here ... 
"""
from transformers import AutoConfig, AutoTokenizer
from bert_qa_model_squad import BertQAModelSquad


def create_model(args):
    args.model_type = args.model_type.lower()
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model_class = None
    if args.squad_model == 'bert_qa_model_squad':
        print('creating bert base model')
        model_class = BertQAModelSquad
    elif args.squad_model == 'xx':
        pass
        # model_class = xx
    else:
        print('no model defined as ', args.squad_model)

    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    return model, config, tokenizer
