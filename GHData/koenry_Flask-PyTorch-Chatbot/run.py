from flask import Flask, render_template, flash, request, jsonify, json
import json, requests,  sqlite3, random
from datetime import datetime
from urllib.parse import quote
from numpy import *
from chat import chat
from urllib.parse import quote
from flask_cors import CORS, cross_origin
import os
from flask import send_from_directory

arrayOfWrongChoices = ["Might need to rephrase that", "Hey I know the answer to this! 42304... No ?", "Try asking something else", "How about NO"]
google = 'google'
app = Flask(__name__)
CORS(app, support_credentials=True)

@app.route('/')
@cross_origin(supports_credentials=True)
def home():
    return render_template('index.html')
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                          'favicon.ico',mimetype='image/vnd.microsoft.icon')

@app.route('/postmethod', methods = ['GET', 'POST'])
@cross_origin(supports_credentials=True)
def getPost():
    
    getPost.jsdata = request.get_json('javascriptData') ### get_json not get form 
    
    return jsonify(getPost.jsdata)
    
@app.route('/getpythondata', methods = ['GET', 'POST'])
@cross_origin(supports_credentials=True)
def pyData():
    sentence = getPost.jsdata
    chat(sentence)
    date = datetime.today().strftime('%Y-%m-%d %H:%M:')
    
    if chat.prob.item() > 0.75:
        con = sqlite3.connect('ChatBot.db')
        cur = con.cursor()
        cur.execute("insert into Chatbot values(?,?,?)",(date, sentence, 'T')) 
        con.commit()
        con.close()
        return json.dumps(chat.answerBot) 
    else:
        if  sentence.lower().startswith(google):
            return json.dumps('Googling... ')
        url = requests.get(f'https://api.duckduckgo.com/?q={sentence}&format=json&pretty=1')
        text = url.text
        y = json.loads(text)
        try:
            if (y["Abstract"] == ""):
                con = sqlite3.connect('ChatBot.db')
                cur = con.cursor()
                cur.execute("insert into Chatbot values(?,?,?)",(date, sentence, 'F')) 
                con.commit()
                con.close()
                return json.dumps("duckduckGo: "+y["RelatedTopics"][0]["Text"])
            else:
                con = sqlite3.connect('ChatBot.db')
                cur = con.cursor()
                cur.execute("insert into Chatbot values(?,?,?)",(date, sentence, 'F')) 
                con.commit()
                con.close()
                return json.dumps("duckduckGo: "+y["Abstract"])
        
        except (IndexError, TypeError):
            con = sqlite3.connect('ChatBot.db')
            cur = con.cursor()
            cur.execute("insert into Chatbot values(?,?,?)",(date, sentence, 'F')) 
            con.commit()
            con.close()
            return json.dumps(random.choice(arrayOfWrongChoices))
            
if __name__ == '__main__':
    app.run( debug = True)
   







    
    
