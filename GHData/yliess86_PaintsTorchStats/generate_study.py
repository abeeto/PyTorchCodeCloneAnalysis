import plotly.figure_factory as ff
import plotly.graph_objs as go
import plotly.plotly as py
import plotly.io as pio
import pandas as pd
import numpy as np
import json
import os

from plotly.offline import plot

STUDY = pd.read_csv('./data/StudyDataset/study.csv', ';', names=('UUID', 'file_name', 'model', 'rate'))

DATA   = []
MODELS = ['PaperRS', 'CustomSS', 'CustomSD']

print(f'Users: {STUDY.UUID.nunique()}')
print(f'Rates: {len(STUDY)}/{STUDY.UUID.nunique() * 480}')
print()

for model in MODELS:
    STD  = STUDY[STUDY.model == model].rate.std()
    MEAN = STUDY[STUDY.model == model].rate.mean()
    
    print(f'Model: {model}', f'Mean: {MEAN:.2f}', f'Std: {STD:.3f}')

for model in MODELS:
    DATA.append([
        len(STUDY[(STUDY.model == model) & (STUDY.rate == rate)]) 
        for rate in range(1, 6)
    ])
    
fig  = ff.create_annotated_heatmap(
    z               = DATA, 
    x               = list(range(1, 6)), 
    y               = MODELS, 
    colorscale      = 'Viridis', 
    annotation_text = DATA,
    showscale       = True,
)

fig['layout']['xaxis']['title']      = 'MOS'
fig['layout']['xaxis']['automargin'] = True
fig['layout']['yaxis']['title']      = 'Model'
fig['layout']['yaxis']['automargin'] = True
fig['layout']['margin']['l']         = 100

pio.write_image(fig, 'heatmap.eps')
pio.write_image(fig, 'heatmap.png')