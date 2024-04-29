import json 
import os


# base path of your website https://xyc.com/ it will be resolved to https://xyc.com/<document_name>#page_no=<page> by the bot
base_path = "file:///d:/aibot/documents/"

def question_cleaner(msg):

    # all question key words are filtered here 
    kwargs = [
        "how to",
        "how can i",
        "how",
        "i want to know about",
        "can you tall me about",
        "i want to go to"
    ]

    for kwarg in kwargs: msg.replace(kwarg,"")

    return msg

def parse_document(message):
    message = question_cleaner(message.lower().strip())
    matches = []
    f = open("document.json")
    documents_data = json.load(f)

    for filename,content in documents_data.items():
        for title,page_no in content.items():
            title = title.lower()
            if message in title or title in message:
                url = os.path.join(base_path,filename,f"#page={page_no}")
                matches.append({"line":title,"url":url})
                url=None

    print(matches)
    return matches
            
