import sys
from notebook.auth import passwd

with open(f"{sys.argv[1]}/.jupyter/.jupyter_password") as f:
    # password should be set in the first line of <repo_dir>/.jupyter_password
    password = f.read().split("\n")[0]

with open(f'{sys.argv[1]}/.jupyter/jupyter_notebook_config.json', 'w') as f:
    f.write('{"NotebookApp": {"password": "%s"}}' % (passwd(password)))
