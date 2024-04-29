import tensorrt as trt
import torch
import transformers
import os
import numpy as np
import timeit


def timeGraph(context, buffers, stream, num_loops=100):
    print("Warm up ...")
    with torch.no_grad():
        for _ in range(20):
            context.execute_async_v2(buffers, stream.cuda_stream)
            stream.synchronize()

    stream.synchronize()

    print("Start timing ...")
    timings = []
    with torch.no_grad():
        for i in range(num_loops):
            start_time = timeit.default_timer()
            context.execute_async_v2(buffers, stream.cuda_stream)
            stream.synchronize()
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
    batch_size = 16

    model = transformers.BertModel(transformers.BertConfig(return_dict=False))
    input_ids = torch.ones(batch_size, 512).type(torch.int32)
    attention_mask = torch.ones(batch_size, 512).type(torch.int32)
    token_type_ids = torch.ones(batch_size, 512).type(torch.int32)

    logger = trt.Logger(trt.Logger.WARNING)

    if '_bert.trt' not in os.listdir('.'):
        torch.onnx.export(model, (input_ids, attention_mask, token_type_ids), '_bert.onnx', input_names=['input_ids', 'attention_mask', 'token_type_ids'], output_names=['last_hidden_state', 'pooler_output'], export_params=True)

        builder = trt.Builder(logger)

        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        parser.parse_from_file('_bert.onnx')

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 31)

        serialized_engine = builder.build_serialized_network(network, config)
        with open('_bert.trt', 'wb') as f:
            f.write(serialized_engine)
    else:
        with open('_bert.trt', 'rb') as f:
            serialized_engine = f.read()

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    context = engine.create_execution_context()

    device = torch.device('cuda:0')

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    token_type_ids = token_type_ids.to(device)

    outputs = (torch.empty(batch_size, 512, 768, dtype=torch.float32).to(device), torch.empty(batch_size, 768, dtype=torch.float32).to(device))
    buffers = [input_ids.data_ptr(), attention_mask.data_ptr(), token_type_ids.data_ptr(), outputs[0].data_ptr(), outputs[1].data_ptr()]

    stream = torch.cuda.Stream()
    timings = timeGraph(context, buffers, stream)
    printStats('onnx to tensorrt bert', timings, batch_size)
