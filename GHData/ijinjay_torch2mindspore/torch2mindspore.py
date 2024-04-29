import torch
import mindspore as ms
from copy import copy
import numpy as np
import io
from collections import defaultdict
# import graphviz

# set mindspore context
ms.context.set_context(mode=ms.context.GRAPH_MODE, device_target="GPU")
# save graph ir to view fusion detail.
# ms.context.set_context(save_graphs=True)
# enable graph kernel optimization.
ms.context.set_context(enable_graph_kernel=True)


# UTILITY FUNCTIONS


def ms_version():
    return ms.__version__


def torch_dtype_to_mindspore(dtype):
    if ms_version() >= '1.0.1' and dtype == torch.bool:
        return ms.bool_
    elif dtype == torch.int:
        return ms.int_
    elif dtype == torch.int8:
        return ms.int8
    elif dtype == torch.int16:
        return ms.int16
    elif dtype == torch.int32:
        return ms.int32
    elif dtype == torch.int64: # TODO
        return ms.int32
    elif dtype == torch.float:
        return ms.float_
    elif dtype == torch.float16:
        return ms.float16
    elif dtype == torch.float32:
        return ms.float32
    elif dtype == torch.float64: # TODO
        return ms.float32
    else:
        raise TypeError("%s is not supported by mindspore" % dtype)


def torch_dtype_from_mindspore(dtype):
    if dtype == ms.int8:
        return ms.int8
    elif dtype == ms.int_:
        return torch.int
    elif ms_version() >= '1.0.1' and dtype == ms.bool_:
        return torch.bool
    elif dtype == ms.int16:
        return torch.int16
    elif dtype == ms.int32:
        return torch.int32
    elif dtype == ms.int64:
        return torch.int64
    elif dtype == ms.float_:
        return torch.float
    elif dtype == ms.float16:
        return torch.float16
    elif dtype == ms.float32:
        return torch.float32
    elif dtype == ms.float64:
        return torch.float64
    else:
        raise TypeError("%s is not supported by torch" % dtype)


def check_torch_dtype(*tensors):
    dtype = None
    for t in tensors:
        if isinstance(t, torch.Tensor):
            if dtype is None:
                dtype = t.dtype
            else:
                assert dtype == t.dtype  # , 'Tensor data types must match')
    assert (
        dtype is not None
    )  # , 'Data type could not be inferred from any item in list')
    return dtype


def add_missing_ms_tensors(network, tensors):
    """Creates missing TensorRT tensors as constants and attaches them to the Torch Tensors"""
    ms_tensors = [None] * len(tensors)

    dtype = check_torch_dtype(*tensors)

    for i, t in enumerate(tensors):
        ms_tensor = None

        # GET MS TENSOR (OR CREATE MS CONSTANT)

        # get tensor w/ _ms_tensor
        # or... add constant for scalar primitive
        if isinstance(t, float) or isinstance(t, int):
            shape = (1,)
            scalar = t * torch.ones(shape, dtype=dtype).cpu().numpy()
            # ms_tensor = network.add_constant(shape, scalar).get_output(0)
            if isinstance(t, float):
                scalar = scalar.astype(np.float32)
            if isinstance(t, int):
                scalar = scalar.astype(np.int32)
            _tensor = ms.Tensor(scalar)
            key = network.add_node(_tensor)
            ms_tensor = key
        elif hasattr(t, "_ms_tensor"):
            ms_tensor = t._ms_tensor

        # or... add constant for leaf tensor w/o _ms_tensor
        else:

            # remove all preceding ones, these can be re-inserted later when broadcasting
            num_preceding_ones = 0
            for j in range(len(t.shape)):
                if int(t.shape[j]) == 1:
                    num_preceding_ones += 1
                else:
                    break
            shape = tuple(t.shape[num_preceding_ones:])

            weight = t.detach().cpu().numpy()

            weight = weight.astype(np.float32)
            _ms_tensor = ms.Tensor(weight)
            key = network.add_node(_ms_tensor)
            t._ms_tensor = key
            ms_tensor = key


        assert ms_tensor is not None

        ms_tensors[i] = ms_tensor

    return ms_tensors


class _MsExpand0(ms.nn.Cell):
    def __init__(self):
        super(_MsExpand0, self).__init__()
    def construct(self, x):
        return ms.ops.expand_dims(x, 0)

def broadcast_ms_tensors(network, ms_tensors, broadcast_ndim):
    """Broadcast TensorRT tensors to the specified dimension by pre-padding shape 1 dims"""
    broadcasted_ms_tensors = [None] * len(ms_tensors)

    for i, t in enumerate(ms_tensors):

        tensor = network.nodes[t]
        if len(tensor.shape) < broadcast_ndim:
            # append 1 size dims to front
            diff = broadcast_ndim - len(tensor.shape)
            shape = tuple([1] * diff + list(tensor.shape))

            # TODO, check print
            ms_cell = _MsExpand0()
            out = ms_cell(tensor)

            op_key = network.add_ops(ms_cell)
            ms_tensor = network.add_node(out)

            network.add_pre(op_key, t)
            network.add_out(op_key, [ms_tensor])
            # layer = network.add_shuffle(t)
            # layer.reshape_dims = shape
            # ms_tensor = layer.get_output(0)
        else:
            ms_tensor = t

        broadcasted_ms_tensors[i] = ms_tensor

    return broadcasted_ms_tensors


# CONVERSION REGISTRY AND HOOKS


CONVERTERS = {}


def get_arg(ctx, name, pos, default):
    if name in ctx.method_kwargs:
        return ctx.method_kwargs[name]
    elif len(ctx.method_args) > pos:
        return ctx.method_args[pos]
    else:
        return default


def attach_converter(ctx, method, converter, method_str):
    """Gets a function that executes PyTorch method and Mindspore converter"""
    global DUMMY_CONVERTERS

    def wrapper(*args, **kwargs):
        skip = True

        # check if another (parent) converter has lock
        if not ctx.lock:
            if converter["is_real"]:
                ctx.lock = True  # only real converters can acquire lock
            skip = False

        # run original method
        outputs = method(*args, **kwargs)

        if not skip:
            ctx.method_args = args
            ctx.method_kwargs = kwargs
            ctx.method_return = outputs
            ctx.method_str = method_str

            # print(converter, type(converter))

            converter["converter"](ctx)

            # convert to None so conversion will fail for unsupported layers
            ctx.method_args = None
            ctx.method_kwargs = None
            ctx.method_return = None
            ctx.lock = False

        return outputs

    return wrapper


class ConversionHook(object):
    """Attaches TensorRT converter to PyTorch method call"""

    def __init__(self, ctx, method, converter):
        self.ctx = ctx
        self.method_str = method
        self.converter = converter

    def _set_method(self, method):
        exec("%s = method" % self.method_str)

    def __enter__(self):
        try:
            self.method_impl = eval(self.method_str)
        except AttributeError:
            self.method_impl = None

        if self.method_impl:
            self._set_method(
                attach_converter(
                    self.ctx, self.method_impl, self.converter, self.method_str
                )
            )

    def __exit__(self, type, val, tb):
        if self.method_impl:
            self._set_method(self.method_impl)

def default_input_names(num_inputs):
    return ["input_%d" % i for i in range(num_inputs)]

def default_output_names(num_outputs):
    return ["output_%d" % i for i in range(num_outputs)]


class LayerNamingNetworkWrapper(object):
    def __init__(self, ctx, network=None):
        self._ctx = ctx
        self._network = network
        self._layer_counts = defaultdict(lambda: 0)

    def _set_layer_name(self, layer):
        def arg_str(arg):
            if isinstance(arg, torch.Tensor):
                return "tensor(shape=%s, dtype=%s)" % (str(list(arg.shape)), str(arg.dtype))
            return str(arg)

        self._layer_counts[layer.type.name] += 1
        args = [arg_str(arg) for arg in self._ctx.method_args]
        kwargs = ["%s=%s" % (key, arg_str(arg)) for key, arg in self._ctx.method_kwargs.items()]
        layer.name = "[%s #%d] %s(%s)" % (layer.type.name, self._layer_counts[layer.type.name],
                                          self._ctx.method_str, ", ".join(args + kwargs))

    def __getattr__(self, name):
        # 通过属性名获取cell
        attr = getattr(self._network, name)
        if callable(attr):
            def wrapper(*args, **kwargs):
                ret = attr(*args, **kwargs)
                if isinstance(ret, ms.nn.Cell):
                    self._set_layer_name(ret)
                return ret

            return wrapper
        else:
            return attr


class ConversionContext(object):
    def __init__(self, network=None, converters=CONVERTERS):
        self.network = LayerNamingNetworkWrapper(self, network)
        self.lock = False
        self.method_args = None
        self.method_kwargs = None
        self.method_return = None
        self.hooks = [
            ConversionHook(self, method, converter)
            for method, converter in converters.items()
        ]

    def __enter__(self):
        for hook in self.hooks:
            hook.__enter__()
        return self

    def __exit__(self, type, val, tb):
        for hook in self.hooks:
            hook.__exit__(type, val, tb)

    def add_inputs(self, torch_inputs, names=None):
        if names is None:
            names = default_input_names(len(torch_inputs))
        self.input_names = names

        for i, torch_input in enumerate(torch_inputs):
            # 有_ms_tensor属性的可以认为是已经作为一个节点了
            if not hasattr(torch_input, "_ms_tensor"):
                dtype = torch_dtype_to_mindspore(torch_input.dtype)
                np_tensor = np.zeros(torch_input.shape)
                if np_tensor.dtype == np.float64:
                    np_tensor = np_tensor.astype(np.float32)
                if np_tensor.dtype == np.int64:
                    np_tensor = np_tensor.astype(np.int32)

                ms_tensor = ms.Tensor(np_tensor)
                ms_tensor_key = self.network.add_node(ms_tensor)
                torch_input._ms_tensor = ms_tensor_key
                self.network.mark_inputs(ms_tensor_key, names[i])

    def mark_outputs(self, torch_outputs, names=None):
        if names is None:
            names = default_output_names(len(torch_outputs))
        self.output_names = names
        for i, torch_output in enumerate(torch_outputs):
            ms_tensor = torch_output._ms_tensor
            self.network.mark_outputs(ms_tensor, names[i])


class MindSporeModule(torch.nn.Module):
    def __init__(self, ms_net=None, input_names=None, output_names=None):
        super(MindSporeModule, self).__init__()
        self._register_state_dict_hook(MindSporeModule._on_state_dict)
        self.net = ms_net
        # self.net.cell_list = list(self.net._cells.values())
        self.input_names = input_names
        self.output_names = output_names

    def _on_state_dict(self, state_dict, prefix, local_metadata):
        state_dict[prefix + "input_names"] = self.input_names
        state_dict[prefix + "output_names"] = self.output_names

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):

        self.input_names = state_dict[prefix + "input_names"]
        self.output_names = state_dict[prefix + "output_names"]

    def forward(self, *inputs):

        for a_node in self.net.sorted_nodes:
            if a_node in self.net.inputs:
                real_node = self.net.inputs[self.net.inputs.index(a_node) + 1]
                name_index = self.input_names.index(real_node)
                np_input = inputs[name_index].detach().cpu().numpy()
                if np_input.dtype == np.int64:
                    np_input = np_input.astype(np.int32)
                if np_input.dtype == np.float64:
                    np_input = np_input.astype(np.float32)
                self.net.nodes[a_node] = ms.Tensor(np_input)
            elif a_node.startswith('ops'):
                ops_inputs = [ self.net.nodes[a_pre] for a_pre in self.net.pre_edges[a_node] ]
                ms_outputs = self.net.ops[a_node](*ops_inputs)

                if len(self.net.out_edges[a_node]) == 1:
                    self.net.nodes[self.net.out_edges[a_node][0]] = ms_outputs
                else:
                    for i in range(len(self.net.out_edges[a_node])):
                        self.net.nodes[self.net.out_edges[a_node][i]] = ms_outputs[i]
            else:
                continue


        outputs = tuple([torch.Tensor(self.net.nodes[_].asnumpy()).cuda() for _ in self.net.outputs[::2]])

        if len(outputs) == 1:
            outputs = outputs[0]
        return outputs

# Driver Code

class JCell(object):
    def __init__(self):
        super(JCell, self).__init__()
        self.nodes = {}
        self.ops = {}
        self.pre_edges = {}
        self.out_edges = {}
        self.inputs = []
        self.outputs = []
        self.sorted_nodes = []

    def mark_inputs(self, key, name):
        self.inputs.append(key)
        self.inputs.append(name)

    def mark_outputs(self, key, name):
        self.outputs.append(key)
        self.outputs.append(name)
        # self.outputs.append((key, name))

    def add_node(self, value, key=None):
        if key is None:
            key = "node" + str(len(self.nodes) + 1)
        self.nodes[key] = value
        return key

    def add_ops(self, value, key=None):
        if key is None:
            key = "ops" + str(len(self.ops) + 1)
        self.ops[key] = value
        return key

    def add_pre(self, ops, nodes):
        self.pre_edges[ops] = nodes

    def add_out(self, ops, nodes):
        self.out_edges[ops] = nodes

    def build_graph(self):
        nodes = self.nodes.keys()
        ops = self.ops.keys()
        verticles = list(nodes) + list(ops)
        self.sorted_nodes = verticles

        # dot = graphviz.Digraph(format='png', comment="Network")
        # for a_node in self.sorted_nodes:
        #     dot.node(a_node)
        # for k, v in self.pre_edges.items():
        #     end = k
        #     for a_start in v:
        #         dot.edge(a_start, k)

        # for k, v in self.out_edges.items():
        #     start = k
        #     for a_end in v:
        #         dot.edge(start, a_end)
        # dot.render()


def torch2mindspore(module, inputs, input_names=None, output_names=None, max_batch_size=1, max_workspace_size=1<25):
    inputs_in = inputs
    inputs = [tensor.clone()[0:1] for tensor in inputs]  # only run single entry

    if isinstance(inputs, list):
        inputs = tuple(inputs)
    if not isinstance(inputs, tuple):
        inputs = (inputs,)

    # run once to get num outputs
    outputs = module(*inputs)
    if not isinstance(outputs, tuple) and not isinstance(outputs, list):
        outputs = (outputs,)

    if input_names is None:
        input_names = default_input_names(len(inputs))
    if output_names is None:
        output_names = default_output_names(len(outputs))

    network = JCell()
    with ConversionContext(network) as ctx:
        ctx.add_inputs(inputs, input_names)
        outputs = module(*inputs)
        if not isinstance(outputs, tuple) and not isinstance(outputs, list):
            outputs = (outputs,)
        ctx.mark_outputs(outputs, output_names)


    network.build_graph()


    module_ms = MindSporeModule(network, input_names=input_names,output_names=output_names)
    return module_ms



# DEFINE ALL CONVERSION FUNCTIONS


def mindspore_converter(method, is_real=True, enabled=True):

    def register_converter(converter):
        CONVERTERS[method] = {"converter": converter, "is_real": is_real}
        return converter

    def pass_converter(converter):
        return converter

    if enabled:
        return register_converter
    else:
        return pass_converter

    return register_converter
