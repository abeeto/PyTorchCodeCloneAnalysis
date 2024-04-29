# main.py
import os

from flask import Blueprint, render_template, flash, request
from flask_login import login_required, current_user
from io import BytesIO
from werkzeug.utils import secure_filename, redirect

from . import app
from .CatDog import predict
from .app import ALLOWED_EXTENSIONS, allowed_file, app

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/model', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        print('post in profile')
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file and allowed_file(file.filename):
            print('file up')
            filename = secure_filename(file.filename)
            image = BytesIO(file.stream.read())
            output = predict.predict(image)
            filepath = os.path.join(os.path.abspath(app.config['UPLOAD_FOLDER']), filename)
            file.stream.seek(0)
            file.save(filepath)
            print('upload_image filename: ' + filename)
            flash(f"Класс:"
                  f" {output.get('class')}, вероятность: {output.get('confidence')}")
            return render_template('again_pic.html', filename='uploads/'+filename)
    return render_template('profile.html', name=current_user.name)

@main.route('/home')
@login_required
def home():
    return render_template('home.html', name=current_user.name)
