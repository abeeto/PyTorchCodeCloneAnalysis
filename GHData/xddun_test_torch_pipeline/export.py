import time
from pathlib import Path

import onnx
import onnxsim
import torch

from modelx import Net


def file_size(path):
    # Return file/dir size (MB)
    mb = 1 << 20  # bytes to MiB (1024 ** 2)
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / mb
    elif path.is_dir():
        return sum(f.stat().st_size for f in path.glob('**/*') if f.is_file()) / mb
    else:
        return 0.0


def export_onnx(model, im, file, opset, train, dynamic, simplify):
    #  ONNX export
    try:
        print(f"starting export with onnx {onnx.__version__}...")

        f = file.with_suffix('.onnx')

        torch.onnx.export(
            model.cpu() if dynamic else model,  # --dynamic only compatible with cpu
            im.cpu() if dynamic else im,
            f,
            verbose=True,
            opset_version=opset,
            training=torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,
            do_constant_folding=not train,
            input_names=['images'],
            output_names=['output'],
            dynamic_axes={
                'images': {
                    0: 'batch',
                    2: 'height',
                    3: 'width'},  # shape(1,3,640,640)
                'output': {
                    0: 'batch',
                    1: 'anchors'}  # shape(1,25200,85)
            } if dynamic else None)

        # Checks
        model_onnx = onnx.load(f)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # Metadata
        onnx.save(model_onnx, f)

        # Simplify
        if simplify:
            try:

                print(f'simplifying with onnx-simplifier {onnxsim.__version__}...')
                model_onnx, check = onnxsim.simplify(model_onnx,
                                                     dynamic_input_shape=dynamic,
                                                     input_shapes={'images': list(im.shape)} if dynamic else None)
                assert check, 'assert check failed'
                onnx.save(model_onnx, f)
            except Exception as e:
                print(f'onnx simplifier failure: {e}')
        print(f'onnx export success, saved as {f} ({file_size(f):.1f} MB)')
        return f
    except Exception as e:
        print(f'onnx export failure: {e}')


def export_engine(model, im, file, train, half, simplify, workspace=4, verbose=True):
    #  TensorRT export https://developer.nvidia.com/tensorrt
    try:
        assert im.device.type != 'cpu', 'export running on CPU but must be on GPU, i.e. `python export.py --device 0`'

        import tensorrt as trt

        if trt.__version__[0] == '7':  # TensorRT 7 handling https://github.com/ultralytics/yolov5/issues/6012
            export_onnx(model, im, file, 12, train, False, simplify)  # opset 12
        else:  # TensorRT >= 8
            # require tensorrt>=8.0.0
            export_onnx(model, im, file, 13, train, False, simplify)  # opset 13
        onnx = file.with_suffix('.onnx')
        assert onnx.exists(), f'failed to export ONNX file: {onnx}'
        f = file.with_suffix('.engine')  # TensorRT engine file
        logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            logger.min_severity = trt.Logger.Severity.VERBOSE

        builder = trt.Builder(logger)
        config = builder.create_builder_config()
        config.max_workspace_size = workspace * 1 << 30
        # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace << 30)  # fix TRT 8.4 deprecation notice

        flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        network = builder.create_network(flag)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(onnx)):
            raise RuntimeError(f'failed to load ONNX file: {onnx}')

        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]
        print(f'TensorRT Network Description:')
        for inp in inputs:
            print(f'TensorRT input "{inp.name}" with shape {inp.shape} and dtype {inp.dtype}')
        for out in outputs:
            print(f'TensorRT output "{out.name}" with shape {out.shape} and dtype {out.dtype}')

        print(f'TensorRT building FP{16 if builder.platform_has_fast_fp16 and half else 32} engine in {f}')
        if builder.platform_has_fast_fp16 and half:
            config.set_flag(trt.BuilderFlag.FP16)
        with builder.build_engine(network, config) as engine, open(f, 'wb') as t:
            t.write(engine.serialize())
        print(f'TensorRT export success, saved as {f} ({file_size(f):.1f} MB)')
        return f
    except Exception as e:
        print(f'TensorRT export failure: {e}')


@torch.no_grad()
def run(
        weights="./model.pth",  # weights path
        imgsz=(28, 28),  # image (height, width)
        batch_size=1,  # batch size
        device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        half=False,  # FP16 half-precision export
        train=False,  # model.train() mode
        dynamic=False,  # ONNX/TF: dynamic axes
        simplify=False,  # ONNX: simplify model
        opset=12,  # ONNX: opset version
        verbose=False,  # TensorRT: verbose log
        workspace=4,  # TensorRT: workspace size (GB)
        include='onnx',  # include formats
):
    t = time.time()
    file = Path(weights)
    if "tensorrt" == include:  # TensorRT require GPU
        device = "cuda:0"
    device = torch.device(device)
    if half:
        assert device.type != 'cpu', '--half only compatible with GPU export, i.e. use --device 0'
        assert not dynamic, '--half not compatible with --dynamic, i.e. use either --half or --dynamic but not both'

    network = Net()
    network.load_state_dict(torch.load('model.pth'))  # load FP32 model
    network.to(device)
    # Checks
    imgsz *= 2 if len(imgsz) == 1 else 1  # expand
    # Input
    im = torch.zeros(batch_size, 1, *imgsz).to(device)  # image size(1,1,H,W) BCHW iDetection

    # Update model
    network.train() if train else network.eval()

    for _ in range(2):
        y = network(im)  # dry runs

    if half:
        im, network = im.half(), network.half()  # to FP16

    shape = tuple(y[0].shape)  # model output shape
    print(f"\nPyTorch starting from {file} with output shape {shape} ({file_size(file):.1f} MB)")

    # Exports
    f = [''] * 10  # exported filenames
    if "tensorrt" == include:  # TensorRT required before ONNX
        f[1] = export_engine(network, im, file, train, half, simplify, workspace, verbose)
    if "onnx" == include:  # OpenVINO requires ONNX
        f[2] = export_onnx(network, im, file, opset, train, dynamic, simplify)

    # Finish
    f = [str(x) for x in f if x]  # filter out '' and None
    if any(f):
        print(f'\nExport complete ({time.time() - t:.2f}s)'
              f"\nVisualize:       https://netron.app")
    return f  # return list of exported files/dirs


if __name__ == '__main__':
    # tensorrt or onnx
    run(include='tensorrt')
