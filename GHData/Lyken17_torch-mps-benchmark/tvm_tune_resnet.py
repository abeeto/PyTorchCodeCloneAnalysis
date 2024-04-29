import os, os.path as osp
import json
import time

import torch
import torch.nn as nn
import torchvision
from torchvision import models

import numpy as np
from copy import deepcopy

import tvm
from tvm import relay, te
from tvm.contrib import graph_executor

def run_tuning(tasks, task_weights, log_file,
    device_key="m1",
    tracker="127.0.0.1",
    port=9190,
    num_measure_trials=5000,
):
    print("Begin tuning...")
    measure_runner = auto_scheduler.RPCRunner(
        device_key,
        tracker,
        port,
        min_repeat_ms=300,
        timeout=30,
        repeat=2
    )
    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=num_measure_trials,
        runner=measure_runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2,
    )
    tuner.tune(tune_option)

from torchvision import models
model = models.resnet50()
model = model.features
x = torch.randn(1, 3, 224, 224)
ts = torch.jit.trace(model, x)

mod, params = relay.frontend.from_pytorch(ts, [("input", x.shape)])

mod = relay.transform.InferType()(mod)
bwd_expr = relay.transform.gradient(mod['main'], mode="first_order")
bwd_mod = tvm.IRModule.from_expr(bwd_expr)
bwd_mod = tvm.transform.Sequential([
    relay.transform.EliminateCommonSubexpr(),
    relay.transform.FoldConstant(),
    relay.transform.FastMath(),
    relay.transform.ToGraphNormalForm(),
])(bwd_mod)

print(bwd_mod)

target="metal"
target_host = "llvm -mcpu=apple-latest -mtriple=arm64-apple-macos"
lib = relay.build(bwd_mod, params=params, target=target, target_host=target_host)

if target == "metal":
    dev = tvm.metal()
else:
    dev = tvm.cpu()

from tvm import auto_scheduler

tasks, task_weights = auto_scheduler.extract_tasks(
    mod["main"], params, target=target, target_host=target_host)
for idx, task in enumerate(tasks):
    print("========== Task %d  (workload key: %s) ==========" %
            (idx, task.workload_key))
    print(task.compute_dag)

log_file = f"lenet_{target}.txt"
run_tuning(tasks, task_weights, log_file)