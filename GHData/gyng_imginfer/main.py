import os

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(), verbose=True)

from app.server.api import init_handlers, make_app  # noqa: E402
from app.server.auth import api_key  # noqa: E402

if api_key:
    print("API_KEY environment variable found; auth required")
else:
    print("API_KEY environment variable not found; auth not required")

init_handlers()

# gunicorn expects flask_app to be exposed,
# so don't wrap this in `if __name__ == "__main__":`
flask_app = make_app(api_key=api_key)

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = os.environ.get("PORT", "8080")
    debug = os.environ.get("DEBUG") is True
    flask_app.run(host=host, port=int(port), debug=debug)
