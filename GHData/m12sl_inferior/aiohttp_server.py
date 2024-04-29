from aiohttp import web
import aiohttp_jinja2
import argparse
import asyncio
import jinja_app_loader
import re
import numpy as np
import cv2
import base64
from uuid import uuid4

from worker import Worker


@aiohttp_jinja2.template("index.html")
async def index(request):
    return dict()


async def handle(request):
    encoded = await request.read()
    prefix = re.match(b"data:image/(.*);base64,", encoded).group(0)
    arr = np.frombuffer(base64.decodestring(encoded[len(prefix):]), np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_ANYCOLOR)
    # ok, here we have img with correct shape
    # let's send it into worker
    out = request.app.worker.process(img)
    # todo: encode result with base64 and send it back, without storing on disk
    fname = "./static/tmp/{}.jpg".format(str(uuid4()))
    cv2.imwrite(fname, out)

    return web.json_response(dict(result=fname))


async def init_app(loop, args):
    app = web.Application(loop=loop)

    aiohttp_jinja2.setup(
        app, loader=jinja_app_loader.Loader(), auto_reload=True, context_processors=[]
    )

    app.router.add_get("/", index)
    app.router.add_post("/handle", handle)

    app.worker = Worker(args.model)
    app.router.add_static("/data", path="./data", name="data")
    app.router.add_static("/static", path="./static", name="static")
    return app


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Listening port, fit current docker-compose by default.",
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
        default='./model.pth',
        help="Path to frozen_graph.pb"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    loop = asyncio.get_event_loop()
    app = loop.run_until_complete(init_app(loop, args))
    web.run_app(app, port=args.port, host=args.host)
