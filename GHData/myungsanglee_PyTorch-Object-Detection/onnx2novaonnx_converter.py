import onnx
#from onnx.tools import update_model_dims
import numpy as np
import onnx.helper as helper
from onnx import shape_inference, TensorProto
from onnx import version_converter
from onnxsim import simplify
from onnx import numpy_helper

import sys
import argparse


ONNX_DTYPE = {
    0: TensorProto.FLOAT,
    1: TensorProto.FLOAT,
    2: TensorProto.UINT8,
    3: TensorProto.INT8,
    4: TensorProto.UINT16,
    5: TensorProto.INT16,
    6: TensorProto.INT32,
    7: TensorProto.INT64,
    8: TensorProto.STRING,
    9: TensorProto.BOOL
}

SUPPORTED_OP_TYPE_LIST = [
'Conv',
'BatchNormalization',
'Relu',
'LeakyRelu',
'PRelu',
'MaxPool',
'AveragePool',
'GlobalMaxPool',
'GlobalAveragePool',
'MaxRoiPool',
'Upsample',
'Split',
'Concat',
'Deconvolution',
'Transpose',
'Reshape',
'Sigmoid',
'Softmax',
'Flatten',
'Tanh',
'Add',
'Mul',
'Max',
'Gemm',
'MatMul',
'Squeeze',
'Unsqueeze',
'Slice',
'Pad',
'ConvTranspose',
'ReduceMean',
'LSTM',
'Abs',
'Clip',
'Min',
'Resize',
'Sub'
]


def process_command():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, required=True, help='input model path')
    parser.add_argument('--output', '-o', type=str, required=True, help='output model path')
    parser.add_argument('--skip_fuse_bn', '-s', type=int, default = 0 ,help='skip bn folding, default = 0(False)')

    return parser.parse_args()


def onnx_attribute_to_dict(onnx_attr):
    #print(onnx_attr)
    if onnx_attr.HasField('name'):
        name = getattr(onnx_attr, 'name')
        #print(name)

    if onnx_attr.HasField('t'):
        return name, numpy_helper.to_array(getattr(onnx_attr, 't'))

    for attr_type in ['f', 'i', 's']:
        if onnx_attr.HasField(attr_type):
            return name, getattr(onnx_attr, attr_type)

    for attr_type in ['floats', 'ints', 'strings']:
        if getattr(onnx_attr, attr_type):
            return name, list(getattr(onnx_attr, attr_type))

def add_input_from_initializer(model : onnx.ModelProto):
    """
    Currently onnx.shape_inference doesn't use the shape of initializers, so add
    that info explicitly as ValueInfoProtos.
    Mutates the model.
    Args:
        model: The ModelProto to update.
    """
    # All (top-level) constants will have ValueInfos before IRv4 as they are all inputs
    if model.ir_version < 4:
        return

    def add_const_value_infos_to_graph(graph : onnx.GraphProto):
        inputs = {i.name for i in graph.input}
        existing_info = {vi.name: vi for vi in graph.input}
        for init in graph.initializer:
            # Check it really is a constant, not an input
            if init.name in inputs:
                continue

            # The details we want to add
            elem_type = init.data_type
            shape = init.dims

            # Get existing or create new value info for this constant
            vi = existing_info.get(init.name)
            if vi is None:
                vi = graph.input.add()
                vi.name = init.name

            # Even though it would be weird, we will not overwrite info even if it doesn't match
            tt = vi.type.tensor_type
            if tt.elem_type == onnx.TensorProto.UNDEFINED:
                tt.elem_type = elem_type
            if not tt.HasField("shape"):
                # Ensure we set an empty list if the const is scalar (zero dims)
                tt.shape.dim.extend([])
                for dim in shape:
                    tt.shape.dim.add().dim_value = dim

        # Handle subgraphs
        for node in graph.node:
            for attr in node.attribute:
                # Ref attrs refer to other attrs, so we don't need to do anything
                if attr.ref_attr_name != "":
                    continue

                if attr.type == onnx.AttributeProto.GRAPH:
                    add_const_value_infos_to_graph(attr.g)
                if attr.type == onnx.AttributeProto.GRAPHS:
                    for g in attr.graphs:
                        add_const_value_infos_to_graph(g)

    return add_const_value_infos_to_graph(model.graph)


def to_nova_onnx(in_model_path, out_model_path, skip_fuse_bn):
    # load model
    onnx_model = onnx.load(in_model_path)
    
    
    # check input shape
    for input in onnx_model.graph.input:
        input_shape = input.type.tensor_type.shape.dim
        for d in input_shape:
            if d.dim_value <= 0:
                assert (False), "Each dimension of input shape must greater than zero, illegal input name = %s"% input.name

    # convert model
    add_input_from_initializer(onnx_model)
    
    # convert model to opset 12
    if onnx_model.opset_import[0].version != 12:
        #assert (False), ": Opset version of the input model is %d, Novaic tool only support Opset version 12."% onnx_model.opset_import[0].version
        print("Warning: Opset version of the input model is {}, Novaic tool support Opset version 12.".format(onnx_model.opset_import[0].version))
        print("Convert from Opset version {} to Opset version 12.".format(onnx_model.opset_import[0].version))
        onnx_model = version_converter.convert_version(onnx_model, 12)
    
    # apply onnx simplify
    model_simp, check = simplify(onnx_model, skip_fuse_bn = skip_fuse_bn)

    assert check, "Simplified ONNX model could not be validated"

    graph = model_simp.graph
    
    for input in graph.input:
        input.name = input.name.replace('/','_')
        input.name = input.name.replace(':','_')
        
    init_name_list = []
    for initializer in graph.initializer:
        initializer.name = initializer.name.replace('/','_')
        initializer.name = initializer.name.replace(':','_')
        init_name_list.append(initializer.name)
        
    for output in graph.output:
        output.name = output.name.replace('/','_')
        output.name = output.name.replace(':','_')
        
    for value_info in graph.value_info:
        value_info.name = value_info.name.replace('/','_')
        value_info.name = value_info.name.replace(':','_')
        
    for node in graph.node:
        node.name = node.name.replace('/','_')
        node.name = node.name.replace(':','_')
        for i in range(len(node.input)):
            node.input[i] = node.input[i].replace('/','_')
            node.input[i] = node.input[i].replace(':','_')
        for i in range(len(node.output)):
            node.output[i] = node.output[i].replace('/','_')
            node.output[i] = node.output[i].replace(':','_')
    
    name_dict = {}
    for i in range(len(graph.node)):
        if graph.node[i].op_type not in SUPPORTED_OP_TYPE_LIST:
            print("Warning: novaic tool can't support ", graph.node[i].op_type)
            
        #modify Conv weight name

        if graph.node[i].op_type == 'Conv':
            if graph.node[i].input[1] in init_name_list:
                name_dict.setdefault(graph.node[i].input[1], graph.node[i].op_type + "_" + graph.node[i].input[1] + "_W")
                graph.node[i].input[1] = graph.node[i].op_type + "_" + graph.node[i].input[1] + "_W"
            if len(graph.node[i].input) > 2:
                if graph.node[i].input[2] in init_name_list:
                    name_dict.setdefault(graph.node[i].input[2],  graph.node[i].op_type + "_" + graph.node[i].input[2] + "_B")
                    graph.node[i].input[2] = graph.node[i].op_type + "_" + graph.node[i].input[2] + "_B"

      
        #modify output tensor_name to (node_name)_Y
        for k in range(len(graph.node[i].input)):
            if graph.node[i].input[k] in name_dict:
                graph.node[i].input[k] = name_dict[graph.node[i].input[k]]
        for l in range(len(graph.node[i].output)):
            name_dict.setdefault(graph.node[i].output[l], graph.node[i].op_type + "_" + graph.node[i].output[l] + "_Y")
            graph.node[i].output[l] = graph.node[i].op_type + "_" + graph.node[i].output[l] + "_Y"

        # Add layer_id attribute for each node
        new_attr = helper.make_attribute("layer_idx", i)
        graph.node[i].attribute.append(new_attr)
        
        #modify Conv weight name
        if graph.node[i].op_type == 'AveragePool' or graph.node[i].op_type == 'MaxPool':
            new_attr = helper.make_attribute("pool_at_pad", 1)
            graph.node[i].attribute.append(new_attr)

    #print(graph.value_info)
    #modify graph output tensor_name to (node_name)_Y
    for m in range(len(graph.output)):
        if graph.output[m].name in name_dict:
            graph.output[m].name = name_dict[graph.output[m].name]
            
    #modify value info name
    for n in range(len(graph.value_info)):
        if graph.value_info[n].name in name_dict:
            graph.value_info[n].name = name_dict[graph.value_info[n].name]

    #modify input name
    for o in range(len(graph.input)):
        if graph.input[o].name in name_dict:
            graph.input[o].name = name_dict[graph.input[o].name] 
            
    #modify initializer name
    for p in range(len(graph.initializer)):
        if graph.initializer[p].name in name_dict:
            graph.initializer[p].name = name_dict[graph.initializer[p].name] 
            
    onnx.save(model_simp, out_model_path)
    print("Convertered NOVA ONNX model done")


if __name__ == '__main__':
    args = process_command()
    print("input:", args.input)
    print("output:", args.output)

    to_nova_onnx(args.input, args.output, args.skip_fuse_bn)

