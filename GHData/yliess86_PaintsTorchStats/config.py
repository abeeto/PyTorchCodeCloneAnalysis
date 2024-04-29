import plotly.io as pio

pio.orca.config.executable = '/usr/local/bin/orca'
pio.orca.config.use_xvfb   = True
pio.orca.config.save()