from modules.prepare_data import *
import sys
import json

if __name__ == "__main__":
    config = json.loads(open('config/config.json').read())

   
    try:
        app = PreparationData(config['pre_data'])
        app.data_processing()

    except KeyboardInterrupt:
        print('MSSG: Process Interrupted...')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    except Exception as e:
        print('PRE PROCESS DATA MSSG: Error {}'.format(e))