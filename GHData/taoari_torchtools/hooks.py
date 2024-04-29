import torch
import collections

class ModuleHook(object):
    def __init__(self):
        super().__init__()
        # NOTE:
        # 1. module can be shared for different inputs and outputs (e.g. nn.ReLU in ResNet)
        # 2. functionals inputs and outputs are not collected (e.g. if using F.relu instead of nn.ReLU)
        # 3. inputs and outputs are not copied, so for an inplace operation, the final results are overwriten
        # Implementaton:
        # v1: use OrderedDict() for .modules, .inputs, .outputs
        # v2: use list of tuple for .modules, .inputs, .outputs
        # v3: use single list of dict
        self._info = [] # use list of dict to be extensable and allow duplicated keys

    def __call__(self, module, inputs, output):
        # NOTE: output = model(*inputs)
        # by default, inputs is tuple of tensors, output is a tensor
        # but inputs can be tuples of abitrary, output can be abitrary.
        # e.g. for MaskRCNN, input or output can be a tensor, int, float, string, None, OrderedDict, or Custom class (ImageList)

        # assert module.__name__ not in self.modules
        assert isinstance(inputs, tuple) # output = model(*inputs)
        self._info.append({'module_name': module.__name__,
            'module': module,
            'inputs': inputs,
            'output': output})

    def clear(self):
        self._info.clear()

    def __iter__(self):
        for item in self._info:
            yield item

    def register_extra_hook(self, name, func):
        for item in self._info:
            item[name] = func(item['module'], item['inputs'], item['output'])


class register_forward_hooks(object):
    def __init__(self, model, leaf_only=False):
        self.model = model
        self.leaf_only = leaf_only
        self.hook = ModuleHook()

    # def __call__(self, func):
    #     def inner(*args, **kwargs):
    #         with self:
    #             return func(*args, **kwargs)
    #     return inner

    def __enter__(self):
        from .utils import named_modules
        for name, mod in named_modules(self.model, leaf_only=self.leaf_only):
            mod.__name__ = name
            mod.register_forward_hook(self.hook)
        return self.hook

    def __exit__(self, *args):
        from .utils import unregister_all_hooks
        self.hook.clear()
        unregister_all_hooks(self.model)

def _module_type(mod, inputs, output):
    return type(mod).__name__

def _output_shape(mod, inputs, output):
    if torch.is_tensor(output):
        return tuple(output.shape)
    else:
        return () # not collected tensors in custom format

def _param_num(mod, inputs, output):
    if mod._modules:
        return 0  # not collected for non-leaf modules
    return sum([param.numel() for param in mod.parameters()])

def test_register_forward_hooks():
    import torchvision.models as models
    model = models.resnet18()

    with register_forward_hooks(model) as forward:
        inputs = torch.randn(1,3,224,224)
        outputs = model(inputs)
        forward.register_extra_hook('module_type', _module_type)
        forward.register_extra_hook('output_shape', _output_shape)
        forward.register_extra_hook('param_num', _param_num)

        for info in forward:
            print(info['module_name'], info['module_type'], info['output_shape'], info['param_num'])

    from torchtools.utils import named_modules
    for name, mod in named_modules(model, leaf_only=True):
        assert len(mod._forward_hooks) == 0

if __name__ == "__main__":
    test_register_forward_hooks()
