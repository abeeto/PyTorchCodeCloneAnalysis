from os import pipe
from flask import Flask, json, request, redirect, url_for,jsonify
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
from flask_cors import CORS



app = Flask(__name__)
CORS(app)

@app.route("/index/<id>")
def index(id):
   str = ""
   transcript = YouTubeTranscriptApi.get_transcript(id)
   json.dumps(transcript)
   
   for i in transcript:
       str += i['text'] 
    
   if(len(str)!=0):
       return redirect(url_for("sum", transcript = str))
    
   else:
        return "<h1> This is Youtube Transcript Summarizer </h1>"


@app.route("/summary/<transcript>")
def sum(transcript):
    summarization = pipeline("summarization")


    originalText = transcript

    summary_text = summarization(originalText)[0]['summary_text']

    if(summary_text):
        print("Hello world")

    return jsonify({'text' : summary_text})
 
 

