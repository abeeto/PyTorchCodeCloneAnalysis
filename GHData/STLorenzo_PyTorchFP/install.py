import os
import json
from src.general_functions import create_dir

abs_path = os.path.abspath(str(__file__) + "/..")
conf_name = "/config/Project_conf.json"
conf_path = os.path.join(abs_path + conf_name)

try:
    with open(conf_path, 'r') as json_file:
        data = json.load(json_file)
        data["base_path"] = abs_path
        create_dir(data['dirs']['data_dir'])
        create_dir(data['dirs']['doc_dir'])

    with open(conf_path, 'w') as json_file:
        json.dump(data, json_file)

except Exception as e:
    print(f"Error locating {conf_name}")

