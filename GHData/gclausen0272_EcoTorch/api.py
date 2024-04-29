from encodings import utf_8
import flask
from flask import request, jsonify
from flask_cors import CORS, cross_origin
import torch
from subprocess import Popen, check_output
import subprocess
from emissionsCal import *
import json

app = flask.Flask(__name__)
app.config["DEBUG"] = True
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/', methods=['GET'])
@cross_origin()
def home():
    return "<h1>My First Deployed Application</h1><p>This is my first deployed application</p>"

@app.route('/api/v1/text', methods=['GET', 'POST'])
@cross_origin()
def text_call():
    code  = request.data
    code = code.decode("utf-8")
    text_file = open("sample_text.py", "w")
    n = text_file.write(code.replace("\\n","\n"))

    text_file.close()
    num_datapoints  = request.args.get('sample', None)
    epochs  = request.args.get('epochs', None)

    class_num = request.args.get('class',None)
    max_len = request.args.get('maxlen', None)
    
    dataset_file = open("./text_dataset.py", "w")
    line1 = "\n"
    line2 = "\n"
    line3 = f"ds = TextClassificationDataset({num_datapoints}, {max_len}, {class_num})"
    line4 = "\n"
    line5 = "dl = DataLoader(ds, batch_size = 32)"
    line6 = "\n"
    line7 = f"input_size = {(1, 128)}"
    line8 = "\n"
    line9 = "\n"
    dataset_file.writelines([line1, line2, line3, line4, line5, line6, line7, line8, line9])
    dataset_file.close()
    subprocess.call(["python", "cleanup_text.py"])
    subprocess.call(["bash", "text_compiler.sh"])
    subprocess.call(["python", "assembled_text_model.py"])

    result = [calculate(epochs)]
    with open('report.json', 'w') as fp:
        json.dump(result, fp)
    return result

    # return jsonify([code, num_datapoints,class_num, max_len])

@app.route('/api/v1/image', methods=['GET', 'POST'])
@cross_origin()
def img_call():
    code  = str(request.data)

    text_file = open("./sample_vision.py", "w")
    code.replace('\t', ' ')
    code.replace('b\'', '')
    code.replace('\'', '')
    n = text_file.write(code.replace("\\n","\n"))
    new_s = code.replace("\\r\\n","\n")[2:-2]
    """code  = str(request.data)
    
    text_file = open("./sample_vision.py", "w")
    new_s = code.replace("\\r\\n","\n")"""
    #n = text_file.write(new_s)
    #n = text_file.write(code.replace("\\n","\n"))

    text_file.close()
    epochs  = request.args.get('epochs', None)
    num_datapoints  = request.args.get('sample', None)

    class_num = request.args.get('class',None)
    hei = request.args.get('height', None)
    sample  = request.args.get('sample', None)
    wid = request.args.get('width', None)


    dataset_file = open("./vision_dataset.py", "w")
    line1 = "\n"
    line2 = "\n"
    line8 = f"num_classes = {class_num}"
    line3 = f"ds = ImageClassificationDataset({num_datapoints}, {(3, int(hei), int(wid))}, {class_num})"
    line4 = "\n"
    line5 = "dl = DataLoader(ds, batch_size = 32)"
    line6 = "\n"
    line7 = f"input_size = {(3, int(hei), int(wid))}"
    line8 = "\n"
    line9 = "\n"
    dataset_file.writelines([line1, line2, line3, line4, line5, line6, line7, line8, line9])
    dataset_file.close()
    
    subprocess.call(["python", "cleanup_vision.py"])
    subprocess.call(["bash", "compiler.sh"])
    subprocess.call(["python", "assembled_vision_model.py"])
    
    result = [calculate(epochs)]
    with open('report.json', 'w') as fp:
        json.dump(result, fp)

    return result

    # return jsonify([code, num_datapoints,class_num,hei, wid])

    # return jsonify([new_s,sample, epochs,class_num,hei, wid])
    
@app.route('/api/v1/echo', methods=['GET'])
@cross_origin()
def prof_call():
    # Check if an ID was provided as part of the URL.
    # If ID is provided, assign it to a variable.
    # If no ID is provided, display an error in the browser.
    echo = ""
    if 'echo' in request.args:
        echo = (request.args['echo'])
    else:
        return "Error: No id field provided. Please specify an id."

    result = "loud and clear: "+ echo

    # Use the jsonify function from Flask to convert our list of
    # Python dictionaries to the JSON format.
    return jsonify(result)

@app.route('/api/v1/torch', methods=['GET'])
@cross_origin()
def torch_call():

    result = str(torch.version)
    # Use the jsonify function from Flask to convert our list of
    # Python dictionaries to the JSON format.
    return jsonify(result)

@app.route('/')
def index():
    return "<h1>Welcome to our server !!</h1>"

if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(port=5000, debug=True)
