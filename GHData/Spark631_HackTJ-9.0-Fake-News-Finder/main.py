#This creates the flask server

from flask import Flask

app = Flask(__name__)

def create_app():
  app = Flask(__name__)
  app.config["SECRET_KEY"] = "123321123321"

  from views import views
  app.register_blueprint(views, url_prefix="/")  
  return app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True)
