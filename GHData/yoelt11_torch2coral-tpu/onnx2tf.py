import onnx
from onnx_tf.backend import prepare

if __name__ == '__main__':
    
    # load onnx model
    onnx_model = onnx.load("onnx-model/onnx_model.onnx")
    
    # build tensorflow representation
    tf_rep = prepare(onnx_model)

    # export tensorflow model
    tf_rep.export_graph('tensorflow-model/tf_model') # generates -> WARNING:absl:Found untraced functions such as gen_tensor_dict while saving (showing 1 of 1) . can be ignored?

