import ctypes
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def load_plugins():
    ctypes.CDLL('/opt/torch2trt/torch2trt/libtorch2trt.so')

    registry = trt.get_plugin_registry()
    torch2trt_creators = [c for c in registry.plugin_creator_list if c.plugin_namespace == 'torch2trt']
    for c in torch2trt_creators:
        registry.register_creator(c, 'torch2trt')


def get_engine(engine_file_path):
    load_plugins()
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine


if __name__ == "__main__":
    engine = get_engine('/home/model/my_model/1/model.plan')
