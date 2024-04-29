try:
    from modules.trainModel import *
    from datetime import datetime
    import json
    import os
    import sys
except ImportError as e:
    print("{} MSSG: Fail Import Module: {}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), e))
    sys.exit(0)

if __name__ == '__main__':
    config = json.loads(open('config/config.json').read())

    try:
        train = Train(config['model'])
        train.trainer()

    except KeyboardInterrupt:
        print('TRAIN MODEL MSSG: Training Process Interrupted...')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    except Exception as e:
        print('TRAIN MODEL MSSG: Error {}'.format(e))
