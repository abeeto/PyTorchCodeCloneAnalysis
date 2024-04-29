import os

from sanic import Sanic, json

app = Sanic(__name__)
env = os.environ.get("ENV".lower(), "dev")

app.update_config(os.path.join(os.path.dirname(__file__),
                               "api", "config", f"{env}.settings"))

from api.views.image import image_bp
from api.views.user import user_bp

app.blueprint(image_bp)
app.blueprint(user_bp)


@app.get("/")
async def health_check(request):
    return json({"status": "ok"})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)
