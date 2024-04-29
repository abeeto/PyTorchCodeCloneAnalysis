import base64
import datetime
import glob
import os

import dash
import dash_core_components as dcc
import dash_html_components as html
import torch
from dash.dependencies import Input, Output, State
from flask import Flask

app = Flask(__name__)

TIMEOUT = 60
dash_app_home = dash.Dash(__name__, server=app, url_base_pathname='/')
dash_app_home.title = 'Student Programmer'

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

dash_app_home.layout = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select File')
        ]),
        style={
            'width': '90%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=False
    ),
    html.Div(id='output-data-upload'),
])


def generate_result_string(results):
    for i, (im, pred) in enumerate(zip(results.imgs, results.pred)):
        str = f'image {i + 1}/{len(results.pred)}: {im.shape[0]}x{im.shape[1]} '
        if pred.shape[0]:
            for c in pred[:, -1].unique():
                n = (pred[:, -1] == c).sum()  # detections per class
                str += f"{n} {results.names[int(c)]}{'s' * (n > 1)}, "  # add to string
        else:
            str += '(no detections)'
        return str.rstrip(', ')


def fetch_model(contents, filename, date):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5x, custom
    decodeit = open(filename, 'wb')
    decodeit.write(decoded)
    decodeit.close()
    results = model(filename)
    results.save('assets')
    result_string = generate_result_string(results)

    return html.Div([
        html.H5('Filename: ' + filename),
        html.H6('Date time: ' + str(datetime.datetime.fromtimestamp(date))),
        html.Hr(),
        html.Div([
            html.Div([
                html.P('Original Image'),
                html.Img(src=contents)
            ], style={'float': 'left', 'position': 'relative', 'margin-left': '10px'}),
            html.Div([
                html.P('Modeled Image'),
                html.Img(src=dash_app_home.get_asset_url(filename))
            ], style={'float': 'left', 'position': 'relative', 'margin-left': '10px'})
        ]),
        html.Div([
            html.Br(),
            html.Hr(),
            html.H3('Result Summary'),
            html.P(result_string)
        ])
    ], style={'float': 'left', 'position': 'absolute'})


def cleanup():
    lis = glob.glob('assets/*.jpg')
    lis.extend(glob.glob('*.jpg'))
    for im in lis:
        os.remove(im)


@dash_app_home.callback(
    Output('output-data-upload', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    cleanup()
    if list_of_contents is not None:
        li = []
        children = fetch_model(list_of_contents, list_of_names, list_of_dates)
        li.append(children)
        return li


if __name__ == '__main__':
    app.run(debug=True, port=5050)
