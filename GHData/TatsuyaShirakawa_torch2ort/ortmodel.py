import numpy as np
import torch
import onnxruntime as ort


def _tuple(x):
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x, )


def to_numpy(input):
    if isinstance(input, torch.Tensor):
        return input.to('cpu').numpy()
    return np.asarray(input)


class ORTModel(object):
    '''
    This class provides a seemless way to run inference in the PyTorch way with a ort session
    '''
    def __init__(self, onnx_file, sess_options=None, providers=[], output_names=None, run_options=None):
        self.sess = ort.InferenceSession(onnx_file, sess_options=sess_options, providers=providers)
        self.input_names = tuple([_.name for _ in self.sess.get_inputs()])
        self.output_names = output_names
        self.run_options = run_options

    def __call__(self, *inputs):
        outputs = self.sess.run(self.output_names,
                                {name: to_numpy(_) for name, _ in zip(self.input_names, _tuple(inputs))},
                                self.run_options)
        if len(outputs) == 1:
            return outputs[0]
        else:
            return tuple(outputs)
