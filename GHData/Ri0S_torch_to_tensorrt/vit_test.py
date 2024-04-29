import torch
import torch_tensorrt
from transformers import ViTModel, ViTConfig
import timeit
import numpy as np


def timeGraph(model, input_tensor1, num_loops=100):
    print("Warm up ...")
    with torch.no_grad():
        for _ in range(20):
            features = model(input_tensor1)

    torch.cuda.synchronize()

    print("Start timing ...")
    timings = []
    with torch.no_grad():
        for i in range(num_loops):
            start_time = timeit.default_timer()
            features = model(input_tensor1)
            torch.cuda.synchronize()
            end_time = timeit.default_timer()
            timings.append(end_time - start_time)
            # print("Iteration {}: {:.6f} s".format(i, end_time - start_time))

    return timings

def printStats(graphName, timings, batch_size):
    times = np.array(timings)
    steps = len(times)
    speeds = batch_size / times
    time_mean = np.mean(times)
    time_med = np.median(times)
    time_99th = np.percentile(times, 99)
    time_std = np.std(times, ddof=0)
    speed_mean = np.mean(speeds)
    speed_med = np.median(speeds)

    msg = ("\n%s =================================\n"
            "batch size=%d, num iterations=%d\n"
            "  Median text batches/second: %.1f, mean: %.1f\n"
            "  Median latency: %.6f, mean: %.6f, 99th_p: %.6f, std_dev: %.6f\n"
            ) % (graphName,
                batch_size, steps,
                speed_med, speed_mean,
                time_med, time_mean, time_99th, time_std)
    print(msg)


if __name__ == '__main__':
    device = torch.device('cuda:0')

    batch_size = 64

    input_image = torch.randn(batch_size, 3, 224, 224).type(torch.float32).to(device)
    half_input_image = torch.randn(batch_size, 3, 224, 224).type(torch.half).to(device)
    inputs = [torch_tensorrt.Input(shape=[batch_size, 3, 224, 224], dtype=torch.float32)]
    half_inputs = [torch_tensorrt.Input(shape=[batch_size, 3, 224, 224], dtype=torch.half)]

    plain_model = ViTModel(ViTConfig()).eval().to(device)

    ts_model = ViTModel(ViTConfig(torchscript=True)).eval().to(device)
    ts_half_model = ViTModel(ViTConfig(torchscript=True)).half().eval().to(device)

    ts_trace_model = torch.jit.trace(ts_model, [input_image])
    ts_trace_half_model = torch.jit.trace(ts_half_model, [half_input_image])


    torch_tensorrt_model = torch_tensorrt.compile(ts_trace_model,
            inputs=inputs,
            enabled_precisions={torch.float32},
            workspace_size=1 << 31,
            truncate_long_and_double=True)
    torch_tensorrt_half_model = torch_tensorrt.compile(ts_trace_model,
            inputs=inputs,
            enabled_precisions={torch.half},
            workspace_size=1 << 31,
            truncate_long_and_double=True)

    timings = timeGraph(plain_model, input_image)
    printStats('Normal ViT', timings, batch_size)

    timings = timeGraph(plain_model.half(), half_input_image)
    printStats('Half Normal ViT', timings, batch_size)

    timings = timeGraph(ts_model, input_image)
    printStats('Script ViT', timings, batch_size)

    timings = timeGraph(ts_half_model, half_input_image)
    printStats('Half Script ViT', timings, batch_size)

    timings = timeGraph(ts_trace_model, input_image)
    printStats('Traced ViT', timings, batch_size)

    timings = timeGraph(ts_trace_half_model, half_input_image)
    printStats('Half Traced ViT', timings, batch_size)

    timings = timeGraph(torch_tensorrt_model, input_image)
    printStats('trt ViT', timings, batch_size)

    timings = timeGraph(torch_tensorrt_half_model, input_image)
    printStats('Half trt ViT', timings, batch_size)
