import torch
import torch_tensorrt
from transformers import BertForSequenceClassification, BertConfig
import timeit
import numpy as np


def timeGraph(model, input_tensor1, input_tensor2, input_tensor3, num_loops=100):
    print("Warm up ...")
    with torch.no_grad():
        for _ in range(20):
            features = model(input_tensor1, input_tensor2, input_tensor3)

    torch.cuda.synchronize()

    print("Start timing ...")
    timings = []
    with torch.no_grad():
        for i in range(num_loops):
            start_time = timeit.default_timer()
            features = model(input_tensor1, input_tensor2, input_tensor3)
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

    batch_size = 16

    input_ids = torch.randint(0, 30000, (batch_size, 512)).type(torch.int32).to(device)
    attention_mask = torch.ones(batch_size, 512).type(torch.int32).to(device)
    token_type_ids = torch.ones(batch_size, 512).type(torch.int32).to(device)

    inputs = [torch_tensorrt.Input(shape=[batch_size, 512], dtype=torch.int32),
            torch_tensorrt.Input(shape=[batch_size, 512], dtype=torch.int32),
            torch_tensorrt.Input(shape=[batch_size, 512], dtype=torch.int32)]

    plain_model = BertForSequenceClassification(BertConfig()).eval().to(device)

    ts_model = BertForSequenceClassification(BertConfig(torchscript=True)).eval().to(device)
    ts_half_model = BertForSequenceClassification(BertConfig(torchscript=True)).half().eval().to(device)

    ts_trace_model = torch.jit.trace(ts_model, [input_ids, attention_mask, token_type_ids])
    ts_trace_half_model = torch.jit.trace(ts_half_model, [input_ids, attention_mask, token_type_ids])


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

    timings = timeGraph(plain_model, input_ids, attention_mask, token_type_ids)
    printStats('Normal Bert', timings, batch_size)

    timings = timeGraph(plain_model.half(), input_ids, attention_mask, token_type_ids)
    printStats('Half Normal Bert', timings, batch_size)

    timings = timeGraph(ts_model, input_ids, attention_mask, token_type_ids)
    printStats('Script Bert', timings, batch_size)

    timings = timeGraph(ts_half_model, input_ids, attention_mask, token_type_ids)
    printStats('Half Script Bert', timings, batch_size)

    timings = timeGraph(ts_trace_model, input_ids, attention_mask, token_type_ids)
    printStats('Traced Bert', timings, batch_size)

    timings = timeGraph(ts_trace_half_model, input_ids, attention_mask, token_type_ids)
    printStats('Half Traced Bert', timings, batch_size)

    timings = timeGraph(torch_tensorrt_model, input_ids, attention_mask, token_type_ids)
    printStats('trt Bert', timings, batch_size)

    timings = timeGraph(torch_tensorrt_half_model, input_ids, attention_mask, token_type_ids)
    printStats('Half trt Bert', timings, batch_size)
