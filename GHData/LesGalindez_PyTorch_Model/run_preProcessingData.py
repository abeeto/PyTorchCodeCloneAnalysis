from modules.pre_processing_csv import *
import json
import sys

if __name__ == "__main__":
    config = json.loads(open('config/config.json').read())

    try:
        preprocess = PreProcessingData(config['csv'])

        preprocess.data_list()
    
    except KeyboardInterrupt:
        print('PRE PROCESSING DATA MSSG: Process Interrupted...')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    except Exception as e:
        print('PRE PROCESSING DATA MSSG: Error {}'.format(e))

    
