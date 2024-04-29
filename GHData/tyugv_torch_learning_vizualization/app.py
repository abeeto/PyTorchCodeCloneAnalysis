import matplotlib.pyplot as plt
from flask import Flask, request, render_template, redirect, url_for
import pickle

app = Flask(__name__)


def get_data(data, header, arr):
    if header in data:
        try:
            item = float(data[header])
            arr.append(item)

        except ValueError:
            print('get value not as number')


def change_data(data, header, el):
    if header in data:
        try:
            return float(data[header])

        except ValueError:
            print('get value not as number')

    return el


class Learning:

    def __init__(self, params=None):
        if params is None:
            self.params = {'max': [], 'mean': [], 'min': [], 'lr': 0, 'cur_lr': 0}
        else:
            self.params = params
        self.changeLr = False

    def refresh_plot(self):
        fig = plt.figure()
        plt.plot(self.params['max'])
        plt.plot(self.params['mean'])
        plt.plot(self.params['min'])
        lr = app.model.params['cur_lr']
        itr = len(app.model.params['mean'])
        plt.title(f'Loss plots with lr = {lr}, iteration {itr}')
        plt.savefig('static/loss_plot.png', format='png')
        plt.close(fig)

    def refresh_params(self, data):
        get_data(data, 'mean_loss', self.params['mean'])
        get_data(data, 'min_loss', self.params['min'])
        get_data(data, 'max_loss', self.params['max'])
        self.params['lr'] = change_data(data, 'lr', self.params['lr'])
        self.params['cur_lr'] = change_data(data, 'cur_lr', self.params['cur_lr'])
        self.refresh_plot()


@app.after_request
def add_header(r):

    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        app.model
    except AttributeError:
        app.model = Learning()

    if request.method == 'POST':
        data = request.form.to_dict()
        app.model.refresh_params(data)
        if 'lr' in data:
            app.model.changeLr = True
        else:
            if app.model.changeLr:
                app.model.changeLr = False
                return render_template('loss.html', url='static/loss_plot.png'), 200, {'lr': app.model.params['lr']}

    return render_template('loss.html', url='static/loss_plot.png')


@app.route('/new_learning', methods=['POST'])
def create_new_learning():
    try:
        del app.model
    except AttributeError:
        pass
    app.model = Learning()
    app.model.refresh_plot()
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)
