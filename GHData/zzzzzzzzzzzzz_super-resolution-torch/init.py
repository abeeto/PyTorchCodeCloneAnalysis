# coding: utf-8
import json
import os
import hashlib
import sqlite3
import subprocess
import datetime
# TODO: возможно, стоит сделать статус завершился эксперимент или нет
# TODO: добавить предложение о продолжении обучения, если такой эксперимент уже был


class Infrastructure(object):
    """
    The object of such class will initialize all folders, dbs, environment variables...
    It's also decompress all archives to corresponding folders (in future)
    """
    GLOBAL_PATH = os.path.dirname(os.path.abspath(__file__))
    DATETIME_FORMAT_STR = '%Y-%m-%d %H-%M-%S'

    def __init__(self, dbname='db.sqlite3', *args, **kwargs):
        self.conn = sqlite3.connect(os.path.join(self.GLOBAL_PATH, dbname))
        self.snapshots_path = self.create_folder_or_pass('snapshots')
        self.logs_path = self.create_folder_or_pass('logs')
        self.models_path = self.create_folder_or_pass('models')
        self.datasets_path = self.create_folder_or_pass('datasets')
        self.scripts_path = self.create_folder_or_pass('scripts')
        self.transforms_path = self.create_folder_or_pass('transforms')
        self.metrics_path = self.create_folder_or_pass('metrics')
        self.visualizers_path = self.create_folder_or_pass('visualizers')
        self.utils_path = self.create_folder_or_pass('utils')
        self.tests_path = self.create_folder_or_pass('tests')
        self.compress_folders = ['logs', 'snapshots', 'datasets', 'tests']

    def init_structure(self):
        self.create_file_or_pass(os.path.join(self.models_path, '__init__.py'))
        self.create_file_or_pass(os.path.join(self.scripts_path, '__init__.py'))
        self.create_file_or_pass(os.path.join(self.metrics_path, '__init__.py'))
        self.create_file_or_pass(os.path.join(self.transforms_path, '__init__.py'))
        self.create_file_or_pass(os.path.join(self.datasets_path, '__init__.py'))
        self.create_file_or_pass(os.path.join(self.visualizers_path, '__init__.py'))
        self.create_file_or_pass(os.path.join(self.utils_path, '__init__.py'))
        self.create_db_or_pass()
        self.decompress()

    def __del__(self):
        self.conn.close()

    def create_folder_or_pass(self, which):
        path = os.path.join(self.GLOBAL_PATH, which)
        if os.path.exists(path):
            if os.path.isdir(path):
                return path
            else:
                raise Exception("expected folder, got file or something else {}".format(path))
        else:
            os.makedirs(path)
            return path

    def create_file_or_pass(self, which):
        path = os.path.join(self.GLOBAL_PATH, which)
        if os.path.exists(path):
            if os.path.isfile(path):
                return path
            else:
                raise Exception("expected file, got folder or something else {}".format(path))
        else:
            with open(path, 'w') as f:
                pass
            return path

    def create_db_or_pass(self):
        try:
            c = self.conn.cursor()

            c.execute('''
                          CREATE TABLE IF NOT EXISTS
                            datasets
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          name TEXT UNIQUE, 
                          path TEXT, 
                          description TEXT, 
                          additional TEXT,
                          classname TEXT,
                          train_count_files INTEGER, 
                          test_count_files INTEGER)
                      ''')

            c.execute('''
                          CREATE TABLE IF NOT EXISTS
                            models
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          name TEXT UNIQUE, 
                          path TEXT, 
                          description TEXT)
                      ''')

            c.execute('''
                          CREATE TABLE IF NOT EXISTS
                            metrics
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          name TEXT UNIQUE, 
                          description TEXT)
                      ''')

            c.execute('''
                          CREATE TABLE IF NOT EXISTS
                            experiments
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          idmd5 TEXT,
                          model_id INTEGER,
                          dataset_id INTEGER,
                          train_params TEXT, 
                          description TEXT,
                          ended INTEGER DEFAULT 0,
                          dt DATETIME,
                          FOREIGN KEY (model_id) REFERENCES models(id),
                          FOREIGN KEY (dataset_id) REFERENCES datasets(id))
                      ''')

            c.execute('''
                          CREATE TABLE IF NOT EXISTS
                            metric_values
                         (experiment_id INTEGER,
                          metric_id INTEGER,
                          value REAL, 
                          PRIMARY KEY (experiment_id, metric_id),
                          FOREIGN KEY (experiment_id) REFERENCES experiments(id),
                          FOREIGN KEY (metric_id) REFERENCES  metrics(id))
                      ''')

            # Save (commit) the changes
            self.conn.commit()
            return 1
        except sqlite3.Error as e:
            print("Database initialization failed")
            raise e

    def compress_for_git(self):
        def compressor(path):
            to_compress = []
            for elem in os.listdir(path):
                p = os.path.join(path, elem)
                if os.path.isdir(p) and ('__pycache__' not in p) and (elem != '.') and (elem != '..'):
                    to_compress.append(p)
            return to_compress

        for folder in self.compress_folders:
            print("Searching for {} to compress".format(folder))
            will_compress = compressor(folder)
            if will_compress:
                print("compressing {}".format(will_compress))
                subprocess.call('tar -czf {}.tar.gz {}'.format(folder, ' '.join(will_compress)), shell=True)

    def decompress(self):
        # TODO: add consistency check
        for folder in self.compress_folders:
            print("Decompressing {}".format(folder))
            if os.path.exists('{}.tar.gz'.format(folder)):
                subprocess.call('tar -xzf {}.tar.gz'.format(folder, folder), shell=True)

    def init_experiment(self, options):
        """
        Returns unique experiment id, based on experiment arguments.
        :param options: experiment's boot parameters argparse obj
        :return:
        """
        opt_array = []
        for arg in vars(options):
            opt_array.append(arg)

        opt_array = sorted(opt_array)

        str_to_hash = ""
        opt_values = []
        for arg in opt_array:
            opt_values.append(getattr(options, arg))

        experiment_id = hashlib.md5(json.dumps(opt_values).encode('utf-8')).hexdigest()
        experiment_id = options.model + '_' + experiment_id
        c = self.conn.cursor()
        c.execute('''
                     SELECT 
                            id, idmd5, model_id, dataset_id, train_params, description, ended, dt
                     FROM
                          experiments
                     WHERE
                          idmd5 = ?
                  ''', (experiment_id,))

        # TODO: продумать как сделать возобновление эксперимента с последнего снэпшота в случае, когда моделей используется две
        if c.fetchone():
            text = ''
            while (text != 'y') and (text != 'n'):
                text = str(
                    input("It seems like you launched such experiment: {}. Do you want to continue?[y/n]:".format(experiment_id))).lower()
            if text == 'y':
                print("Ok will start this experiment...")
            else:
                print("Then change some start parameters and restart")
                exit(0)
        # init experiment and return experiment id
        cr_time = datetime.datetime.now()
        try:
            c.execute('''
                         INSERT INTO experiments (idmd5, model_id, dataset_id, train_params, description, dt) 
                         SELECT 
                          ?, t1.id as model_id, t2.id as dataset_id, ?, ?, ?
                         FROM
                             (SELECT 
                               id 
                             FROM
                               models
                             WHERE name=?) AS t1,
                             (SELECT 
                               id 
                             FROM
                               datasets
                             WHERE name=?) AS t2
                    
                      ''',
                      (experiment_id,
                       json.dumps(vars(options)),
                       options.description,
                       cr_time,
                       options.model,
                       options.dataset))
            self.conn.commit()
        except sqlite3.Error as e:
            print("Experiment registration failed")
            raise e

        return experiment_id, cr_time

    def init_test(self, options):
        c = self.conn.cursor()
        opt_dict = vars(options)
        if ('experimentId' in opt_dict) and opt_dict['experimentId']:
            try:
                c.execute('''
                            SELECT 
                              id, idmd5, dt
                            FROM
                              experiments
                            WHERE id=?
                          ''',
                          (opt_dict['experimentId'],))
            except sqlite3.Error as e:
                print("Getting experiment md5 id and cr_time failed")
                raise e
        if ('experimentIdMd5' in opt_dict) and opt_dict['experimentIdMd5']:
            try:
                c.execute('''
                            SELECT 
                              id, idmd5, dt
                            FROM
                              experiments
                            WHERE idmd5=?
                          ''',
                          (opt_dict['experimentIdMd5'],))
            except sqlite3.Error as e:
                print("Getting experiment cr_time failed")
                raise e
        try:
            id, idmd5, cr_time = c.fetchone()
            return id, idmd5, cr_time
        except sqlite3.Error as e:
            print("You should provide experimentId or experimentIdMd5 to get model klass to use")
            exit(-1)


if __name__ == '__main__':
    infra = Infrastructure()
    infra.init_structure()
