import tensorflow as tf
import numpy as np

def representative_dataset():
    for b in range(200): 
        yield [tf.dtypes.cast(input_tensor_list, tf.float32)]

    #for data in tf.data.Dataset.from_tensor_slices(input_tensor_list).batch(1).take(100):
    #    yield [tf.dtypes.cast(data, tf.float32)]

if __name__=='__main__':
    
    # generate converter
    # converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model('tensorflow-model/tf_model', 
    #         input_arrays=["serving_default_input"],
    #         output_arrays=["Const"],
    #      ) # or converter = tf.lite.TFLiteConverter.from_keras_model_file(saved_model_dir)
    converter = tf.lite.TFLiteConverter.from_saved_model('tensorflow-model/tf_model',
            signature_keys=["serving_default"], tags=['serve'])

    # pass options to the converter
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # pass representative dataset (the difficult part)
    batch_size, height, width, channel = 100, 64, 256, 3
    # input_tensor_list = [tf.random.uniform((height, width, channel)) for i in range(100)]
    input_tensor_list = tf.random.uniform((batch_size, height, width, channel))
    converter.representative_dataset = representative_dataset

    # suported operations
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    # set input and output types : only available in tf2
    converter.inference_input_type = tf.int8 # or tf.int8
    # Note: tf.float32 and int8 yield to a correct output, while uint8 not?
    converter.inference_output_type = tf.int8 # or tf.int8
    converter.experimental_new_converter = True

    # convert
    tflite_quant_model = converter.convert()

    # save model
    with open('tflite-model/tflite_model.tflite', 'wb') as f:
        f.write(tflite_quant_model)
