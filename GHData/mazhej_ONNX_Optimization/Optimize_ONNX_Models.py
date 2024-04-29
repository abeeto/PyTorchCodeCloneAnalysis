from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import onnx
from onnx import optimizer
import onnx.utils
import onnx
from onnx import helper, shape_inference
from onnx import TensorProto

# Load the onnx model
models_list=['resnet152','resnet50','squeezenet1_1','densenet169','mobilenet_v2','mobilenet_v1_ssd','ResNet101_DUC_HDC',
'resnet50_pruned_70_state_dict.pth','resnet50_pruned_83_state_dict.pth','resnet50_pruned_85_state_dict.pth']
# models_list = ['yolov3','yolov3','yolov3_tiny','yolov3-spp','yolo_best_pt'

for m in models_list:

    model = onnx.load(f'/1TBstorage/OnnxModels_New/{m}.onnx')

    # print('The model before optimization:\n\n{}'.format(onnx.helper.printable_graph(model.graph)))

    #check the model
    onnx.checker.check_model(model)

    # A full list of supported optimization passes can be found using get_available_passes()
    all_passes = optimizer.get_available_passes()

    # Pick one pass as example
    passes = ['eliminate_unused_initializer','fuse_bn_into_conv']

    # polish the model 
    # model = onnx.utils.polish_model(model)

    # Apply the optimization on the original model
    optimized_model = optimizer.optimize(model, passes)
    inferred_model = shape_inference.infer_shapes(optimized_model)
    #save the model
    onnx.save(inferred_model, f'/1TBstorage/OnnxOptimized_New/{m}_Opt.onnx')


    #onnxruntime
    import onnxruntime as rt

    sess_options = rt.SessionOptions()

    # Set graph optimization level
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_BASIC

    # To enable model serialization after graph optimization set this
    sess_options.optimized_model_filepath =  f'/1TBstorage/OnnxOptimized_New/{m}_Opt_Runtime.onnx'

    session = rt.InferenceSession( f'/1TBstorage/OnnxOptimized_New/{m}_Opt.onnx', sess_options)