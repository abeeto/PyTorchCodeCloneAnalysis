'''
Main file that mediates between training and evaluation session as per user's choice.
'''

from main_cfg import ARGS


if ARGS.mode == 'train':
    import train

    train.main()

elif ARGS.mode == 'eval':
    import evaluation

    evaluation.main()
