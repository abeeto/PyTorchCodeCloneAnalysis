from comet_ml import Experiment

experiment = Experiment(
    api_key="kjHZ2QnMoIZGgW9yFH5zPCLkD",
    project_name="general",
    workspace="jerryjack121",
    disabled=False
)
# iter_meter = IterMeter()
with experiment.train():
    for i in range(10):
        experiment.log_metric('loss',i,step=i)
        # iter_meter.step()
