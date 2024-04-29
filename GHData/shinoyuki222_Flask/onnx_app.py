from flask import Flask, jsonify,request, render_template
app = Flask(__name__)
from load_onnx import NLU_module
import argparse
import os
import onnxruntime
from metrics import get_entities

Module = NLU_module()

@app.route('/query', methods=['GET'])
def predict():
    # read test_sentence
    input_sentence = request.args.get('text')
    results = Module.Inference(input_sentence)

    return (results)


@app.route('/form', methods=['GET'])
def formexample():
    # read test_sentence
    input_sentence = request.args.get('text')
    results = Module.Inference(input_sentence)
    return render_template('result.html', input_sentence=results["Input_sentence"], pred_lbls=results["Raw Labels"], pred_cls=results["Intent"], slot=results["Megred Mentions"])

@app.route('/json-example')
def jsonexample():
    return 'Todo...'

if __name__ == '__main__':
    # app.run(debug=True, port=5000, host='0.0.0.0') 
    app.run(debug=True, port=5000) #run app in debug mode on port 5000
