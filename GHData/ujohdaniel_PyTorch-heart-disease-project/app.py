from flask import Flask, render_template, request, redirect
import model

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods= ['POST'])
def predict():


    age = request.form['age']
    sex = request.form['sex']
    cp = request.form['cp']
    trestbps = request.form['trestbps']
    chol = request.form['chol']
    fbs = request.form['fbs']
    restecg = request.form['restecg']
    thalach = request.form['thalach']
    exang = request.form['exang']
    oldpeak = request.form['oldpeak']
    slope = request.form['slope']
    ca = request.form['ca']
    thal = request.form['thal']

    prediction = model.predict_model(
                age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
                )

    if prediction== 1:
        prediction ="positive. See your doctor!"
    else:
        prediction ="negative. You are alright!"

    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
