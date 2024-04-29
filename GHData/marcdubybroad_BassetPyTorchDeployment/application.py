# imports
from flask import Flask, jsonify, request

# create the application
app = Flask(__name__)

@app.route('/test', methods=['GET'])
def test():
    if request.method == 'GET':
        return jsonify({"dude": "hello dude"})

if __name__ == "__main__":
    app.run(debug = True)


