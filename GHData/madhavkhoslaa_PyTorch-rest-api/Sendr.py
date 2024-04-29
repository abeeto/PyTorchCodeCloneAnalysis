import flask
import base64
import json

class Sendr():
    def __init__(self):
        pass

    def Ser(self, File_LocationisText = False, label = None):
        if isText:
            pass
        else:
            #The image is encoded into Base64 so that it can be sent
            #as a json object
            with open("File_Location", "rb") as image_file:
                encoded = base64.b64encode(image_file.read())
                return json.dumps({'image': encoded, 'label': label})

    def DeSer(self, json_obj, save_folder, isText = False):
        if isText:
            pass
        else:
            #If it is a image data it is expected to be a base64 encrp            ted image.
            strr = json.loads(json_obj)
