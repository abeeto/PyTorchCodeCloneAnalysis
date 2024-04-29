# A file to generate compute cababilities out of a list of whl files.
# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pathlib
from collections import OrderedDict
import shutil
import subprocess
from tqdm import tqdm
import csv

output_file = 'output.csv'
WHL_FILE_FOLDER = '/your/folder/with/whl/'


class CudaCompute:

    def __init__(self):
        self.oDi = OrderedDict()

    def get_whl_files(self):

        directory_list = os.listdir(WHL_FILE_FOLDER)
        for filename in tqdm(directory_list):
            if filename.endswith(".whl"):
                file_with_path = os.path.join(WHL_FILE_FOLDER, filename)
                p = pathlib.Path(file_with_path)

                subprocess.run(['unzip', '-d', p.with_suffix('.folder'), file_with_path], stdout=subprocess.DEVNULL)
                directory_list2 = os.listdir(str(p.with_suffix('.folder')) + "/torch/lib/")
                for filename2 in directory_list2:
                    my_command = self.generate_command(p, filename2)
                    # print("Command: " + my_command)

                    myres = str(subprocess.getoutput(my_command))
                    myres = repr(myres)
                    myres = myres.replace("arch = sm_", "", 1).replace("arch = sm_", ", ").replace("\"", "").replace("\'", "")

                    if not myres.endswith("device code"):
                        self.oDi[filename] = myres
                        shutil.rmtree(p.with_suffix('.folder'))
                        break

        print(self.oDi)

    def generate_command(self, p, filename):
        so_file = str(p.with_suffix('.folder'))
        so_file = so_file + "/torch/lib/" + filename
        so_file = so_file + " | grep arch | sort | uniq | tr -d '\\n' "
        return "cuobjdump " + so_file

    def generate_csv(self):
        with open('output.csv', 'w') as f:
            writer = csv.writer(f)
            for key, value in self.oDi.items():
                writer.writerow([key, value])


if __name__ == "__main__":
    obj = CudaCompute()
    obj.get_whl_files()
    obj.generate_csv()
