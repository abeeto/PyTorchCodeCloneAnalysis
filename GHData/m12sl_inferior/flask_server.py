import argparse
import base64
import cv2
import re
from uuid import uuid4
from flask import Flask, request, render_template, jsonify, g
import numpy as np

from worker import Worker

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/handle", methods=["POST"])
def handle():
    encoded = request.data
    prefix = re.match(b"data:image/(.*);base64,", encoded).group(0)
    arr = np.frombuffer(base64.decodestring(encoded[len(prefix):]), np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_ANYCOLOR)
    # ok, here we have img with correct shape
    # let's send it into worker

    # todo: move Worker to global app context
    # todo: get path from config
    worker = Worker("./data/graph.pb")
    out = worker.process(img)
    # todo: encode result with base64 and send it back
    fname = "./static/tmp/{}.jpg".format(str(uuid4()))
    cv2.imwrite(fname, out)

    return jsonify(result=fname)


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Listening port",
    )
    p.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Listening host, use default for visibility from external hosts",
    )
    p.add_argument(
        "--model",
        type=str,
        default='./model.pt',
        help="Path to model.pt"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    app.run(port=args.port, host=args.host, debug=True)
