from flask import Flask, flash, request, redirect, render_template
import os
from io import BytesIO
from werkzeug.utils import secure_filename
from .CatDog import predict
 
app = Flask(__name__, static_url_path='/Flask_HWTrash/static')
 
# UPLOAD_FOLDER = os.path.abspath('./project/static/uploads/')
UPLOAD_FOLDER = 'project/static/uploads/'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif', 'webp'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     
 
@app.route('/', methods=['GET'])
def home():
    return render_template('profile.html')
 
@app.route('/', methods=['POST'])
def upload_image():
    print('enter post /')
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('Не выбрано изображение для загрузки')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        print('file up')
        filename = secure_filename(file.filename)
        image = BytesIO(file.stream.read())
        output = predict.predict(image)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print('upload_image filename: ' + filename)
        flash(output.items())
        return render_template('profile.html', filename=filename)
    else:
        flash('Допустимые типы изображений - png, jpg, jpeg, gif')
        return redirect(request.url)
 
@app.route('/display/<filename>')
def display_image(filename):
    return render_template('profile.html')


if __name__ == "__main__":
    app.run()
