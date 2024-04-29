#importation les méthode dans la page test1
from API_Context_q_r.camombert_q_r__init__ import Q_R
from GPT3_Q_A.gpt3 import GPT3
from WIKI_Q_A.wiki_q_a import Wiki_Q_R
from Object_detection.detector__init__ import Detector
# import main Flask class and request object
from flask import Flask, request, jsonify
import cv2
from PIL import Image
#import flaskrestful

#from flask_restful import reqparse


q_r = Q_R()
q_r.loadModel()


#get gpt3 q
gpt3 = GPT3()
gpt3.connect()
#gpt3.get_gpt3aq()


#get wikipidia  q a
wiki = Wiki_Q_R()



# create the Flask app
app = Flask(__name__)

# allow both GET and POST requests
@app.route('/get/response', methods=['GET', 'POST'])
def form_example():
    # handle the POST request
    if request.method == 'POST':
        data = request.json
        context = data['context']
        question = data['question']
        #context = request.args.get('context')
        #question = request.args.get('question')
        response = q_r.predict(context, question)
        print(' context = ', context, ' question=', question, 'response', response )
        return jsonify({'message_text': data['question'], 'response': response})
    else:
        return jsonify({'message_text': data['question'], 'response': "eurreur"})




"""test avec du score et sentiment"""

# allow both GET and POST requests
@app.route('/get/response+Answer', methods=['GET', 'POST'])
def form_example1():
    # handle the POST request
    if request.method == 'POST':
        context = request.form.get('context')
        question = request.form.get('question')
        answer = request.form.get('answer')
        response = q_r.predict(context,question)
        score = q_r.compute_f1(response,answer)

        return jsonify({'message_text': question,
                        'true_answer' : answer,
                        'response': response,
                        'score':score})
    else:
        return jsonify({'message_text': request.form.get('question'),
                        'true_answer' : request.form.get('answer '),
                        'response': 0,
                        'score':0})



#todo define : a root /translate/gpt3 with method post
# that send a ansewer respence and get a question request in gpt3

# allow both GET and POST requests
@app.route('/wiki/question', methods=['GET', 'POST'])
def getwikiq_a():
    # handle the POST request
    if request.method == 'POST':
        data = request.json
        #context = request.form.get('context')
        question = data['question']
        lang = data['lang']
        # question = request.form.get('question')
        # lang = request.form.get('lang')
        # text=wiki.question_answer1(question,lang)
        try:
            text=wiki.qeustion_answer_paragraph(question,lang)
            response = q_r.predict(text,question)
        #score = q_r.compute_f1(response,answer)


            return jsonify({'question': question,
                            'language': lang,
                            'answer': response
                            })
        except:
            return "erreur"

    else:
        return jsonify({'text': text,
                        'language': lang,
                        'answer': 0
                        })



#todo define : a root /translate/gpt3 with method post
# that send a ansewer respence and get a question request in gpt3

# allow both GET and POST requests
@app.route('/wiki/question/translate', methods=['GET', 'POST'])
def getwikiq_a1():
    # handle the POST request
    if request.method == 'POST':
        data = request.json
        #context = request.form.get('context')
        question = data['question']
        lang = data['lang']

        # question = request.form.get('question')
        # lang = request.form.get('lang')
        # text=wiki.question_answer1(question,lang)
        if lang != 'fr':
            try:
                questiontrad=wiki.tranlatequestion(question,lang)#traduire de n lang a l francais
                text = wiki.qeustion_answer_paragraph1(questiontrad, lang)
                response = q_r.predict(text, questiontrad)
                response1 = wiki.tranlated(response,lang)
                print('question traduire=',questiontrad)
                print('reponce nn traduire=',response)
                return jsonify({'question': question,
                                'language': lang,
                                'answer': response1
                                })

            except:
                return "erreur1"
        else:
            try:
                text = wiki.qeustion_answer_paragraph(question, lang)
                response = q_r.predict(text, question)

                # score = q_r.compute_f1(response,answer)

                return jsonify({'question': question,
                                'language': lang,
                                'answer': response
                                })
            except:
                return "erreur2"

    else:
        return jsonify({'text': text,
                        'language': lang,
                        'answer': 0
                        })



#todo define : a root /get/gpt3 with method post
# that send a ansewer respence and get a question request in gpt3

# allow both GET and POST requests
# @app.route('/get/gpt3', methods=['GET', 'POST'])
# def getgpt3():
#     # handle the POST request
#     if request.method == 'POST':
#         data = request.json
#         #context = request.form.get('context')
#         # question = request.form.get('question')
#         # answer = gpt3.get_gpt3aq(question)
#         question = data['question']
#         answer = gpt3.get_gpt3aq(question)
#         #response = q_r.predict(context,question)
#         #score = q_r.compute_f1(response,answer)
#
#         return jsonify({'message': question,
#                         'response': answer
#                         })
#     else:
#         return jsonify({'message_text': question,
#                         'response': 0
#                         })



#todo define : a root /get/gpt3 with method post
# that send a ansewer respence and get a question request in gpt3

# allow both GET and POST requests
@app.route('/get/gpt3', methods=['GET', 'POST'])
def getgpt3():
    # handle the POST request
    if request.method == 'POST':
        data = request.json
        #context = request.form.get('context')
        # question = request.form.get('question')
        # answer = gpt3.get_gpt3aq(question)
        question = data['question']
        lang = data['lang']
        if lang != 'en':
            answer = gpt3.get_gpt3aq_with_translate(question,lang)
        else:
            answer = gpt3.get_gpt3aq(question)
        #response = q_r.predict(context,question)
        #score = q_r.compute_f1(response,answer)

        return jsonify({'message': question,
                        'response': answer
                        })
    else:
        return jsonify({'message_text': question,
                        'response': 0
                        })



#todo define : a root /translate/gpt3 with method post
# that send a ansewer respence and get a question request in gpt3

# allow both GET and POST requests
@app.route('/translate/gpt3', methods=['GET', 'POST'])
def getgpt3_translate():
    # handle the POST request
    if request.method == 'POST':
        #context = request.form.get('context')
        text = request.form.get('text')
        language = request.form.get('language')
        translated = gpt3.get_gpt3_translate(text,language)
        #response = q_r.predict(context,question)
        #score = q_r.compute_f1(response,answer)


        return jsonify({'text': text,
                        'language': language,
                        'translated': translated
                        })
    else:
        return jsonify({'text': text,
                        'language': language,
                        'translated': 0
                        })








#todo VISUAL PART DETECT
# import methods

#
# classFile = "./Object_detection/coco.names"
# modelURL='http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz'
# modelname= "ssd_mobilenet_v2_320x320_coco17_tpu-8"
# cacheDir = "./Object_detection/pretrained_models"
#
#
# detector = Detector()
# detector.readClasses(classFile)
# detector.downloadModel(modelURL)
# detector.loadModel()
# #detector.loadModeldowloading(modelname,cacheDir)



#todo define : API that returns JSON with classes found in images
@app.route('/detect', methods=['GET', 'POST'])
# def getdetect():
#     #
#     # # handle the POST request
#     # if request.method == 'POST':
#     #     #images = request.files["images"].read()
#     #     images = Image.open(request.files["images"])
#     #
#     #     #images = request.files.getlist("images")
#     #     #response = detector.predictImage(images)
#     #     response = detector.createdetectionBox(images)
#     #
#     #     #response = q_r.predict(context,question)
#     #     #score = q_r.compute_f1(response,answer)
#     #
#     #     try:
#     #         return jsonify({"response": response}), 200
#     #     except FileNotFoundError:
#     #         abort(404)




@app.route('/json-example')
def json_example():
    return 'JSON Object Example'

if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(debug=True, port=5000)