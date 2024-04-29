import os
from random import random, randint

from mlflow import log_metric, log_param, log_artifacts

"""Description
https://github.com/mlflow/mlflow/blob/master/examples/quickstart/mlflow_tracking.py

How to use
  $ python quick_start_mlflow.py
  $ mlflow ui
"""


if __name__ == "__main__":
    print("Running mlflow_tracking.py")

    log_param("param1", randint(0, 100))

    log_metric("foo", random())
    log_metric("foo", random() + 1)
    log_metric("foo", random() + 2)

    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    with open("outputs/test.txt", "w") as f:
        f.write("hello world!")

    log_artifacts("outputs")
