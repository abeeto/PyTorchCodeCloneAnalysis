import os
import argparse
import functools
import time
import torch
import torchvision.models
import multiprocessing

exclude_models = ['mnasnet1_3', 'mnasnet0_75']


def get_gpu_name(device_number):
    return f'{torch.cuda.get_device_name(device_number)}'


def get_gpu_device():
    return torch.cuda.current_device()


def create_test_batch(batch_size):
    return torch.rand(size=(batch_size, 3, 224, 224))


def invoke_model(model_name, dev):
    model = eval(f'torchvision.models.{model_name}(pretrained=True)')
    model.to(dev)
    return model


def run_model_test(model, dev, batch_size, number_of_batches):
    test_data = create_test_batch(batch_size=batch_size).to(dev)
    _ = model(test_data)
    start_time = time.time()
    try:
        for i in range(number_of_batches):
            _ = model(test_data)
        elapsed = time.time() - start_time
        return elapsed
    except RuntimeError as re:
        pass  # eg. GPU OOM

    return 0.


def run_model(model_name, dev, batch_size, number_of_batches):
    try:
        model = invoke_model(model_name, dev)
        return run_model_test(model, dev, batch_size, number_of_batches)
    except ValueError as ve:
        pass
    except NotImplementedError as ne:
        pass  # eg. pre trained not supported
    except RuntimeError as re:
        pass  # eg. Calculated padded input size per channel: ..

    return 0.


def run_model_new_device(model_input):
    dev = get_gpu_device()
    model_name, batch_size, number_of_batches = model_input
    return run_model(model_name, dev, batch_size, number_of_batches)


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output_dir", default='results', help="path to output results")
    ap.add_argument("-f", "--output_filename", help="filename for results")
    ap.add_argument("-n", "--number_of_batches", default=100, type=int, help="number_of_batches")
    ap.add_argument("-s", "--start_batch_size", default=1, type=int, help="start_batch_size 2^n")
    ap.add_argument("-t", "--stop_batch_size", default=10, type=int, help="stop_batch_size 2^n")
    return vars(ap.parse_args())


if __name__ == "__main__":
    args = get_args()
    batch_sizes = list(map(functools.partial(pow, 2), range(args['start_batch_size'], args['stop_batch_size'])))
    # nb must use another process, bc gpu mem cleanup
    pool = multiprocessing.Pool(processes=1)
    if args['output_filename'] is None:
        gpu_name = pool.map(get_gpu_name, [get_gpu_device(),])[0]
        output_filename = os.path.join(args['output_dir'], f'{str(gpu_name).lower().replace(" ", "_")}.md')
        os.makedirs(args['output_dir'], exist_ok=True)
    else:
        output_filename = args['output_filename']

    print(f'results to: {output_filename}')
    f = open(output_filename, 'w')
    elapsed_times_model_batch_size = {}
    models_to_run = []
    for batch_size in batch_sizes:
        for model_name in dir(torchvision.models):
            beginning_letter = model_name[0]
            if beginning_letter != '_' and not beginning_letter.isupper() and model_name not in exclude_models:
                model_class = eval(f'torchvision.models.{model_name}')
                if callable(model_class):
                    models_to_run.append((model_name, batch_size, args['number_of_batches']))

    elapsed_times_per_model = pool.map(run_model_new_device, models_to_run)
    elapsed_times_model_batch_size.update(dict(zip(models_to_run, elapsed_times_per_model)))

    f.write('|batch size|model name|(s)|(fps)|\n')
    f.write('|---|---|---|---|\n')
    for (model_name, batch_size, _), model_elapsed_time in elapsed_times_model_batch_size.items():
        if model_elapsed_time:
            f.write(f'|{batch_size}|{model_name}|{model_elapsed_time:.1f}|{int((batch_size * args["number_of_batches"]) / model_elapsed_time):,}|\n')

    f.close()
