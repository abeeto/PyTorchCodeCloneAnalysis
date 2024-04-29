import json

from flask import Flask, Response, request

app = Flask(__name__)


@app.route("/")
def hello():
    payload = request.args.get('payload')
    print(payload)
    data = [{'name': 'rideOS', 'logo': 'https://res-5.cloudinary.com/crunchbase-production/image/upload/c_lpad,h_256,w_256,f_auto,q_auto:eco/yaekyawvj6e0sqxwnxvf', 'link': 'https://rideos.ai', 'keywords': ['car']},
            {'name': 'Dashblock', 'logo': 'https://ph-files.imgix.net/a6bb9056-fae3-4ff6-bd4d-4913300ffa9b?auto=format', 'link': 'https://dashblock.com',
                'keywords': ['api', 'webscrapper']},
            {'name': 'Netlify', 'logo': 'logo', 'https://www.netlify.com/img/press/logos/logomark.png': 'https://www.netlify.com/', 'keywords': []}]
    return Response(json.dumps(data),  mimetype='application/json')


if __name__ == '__main__':
    app.run(debug=True)
