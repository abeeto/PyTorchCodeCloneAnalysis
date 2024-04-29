from flask import Flask

import passage

app = Flask(__name__)
app.register_blueprint(passage.bp)
app.config.from_mapping(
    SECRET_KEY='dev'
)

if __name__ == '__main__':
    app.run()
