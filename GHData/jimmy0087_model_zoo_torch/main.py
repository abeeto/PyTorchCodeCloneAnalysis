from modelzoo.libs.train import TrainPipline, parse_opts

opt = parse_opts('./modelzoo/config/opts.json')
myTest = TrainPipline(opt)
traData, valData = myTest.datasets('CIFAR10')
myModel = myTest.model(model_name='alexnet')
myTest.train(traData, valData, myModel)


