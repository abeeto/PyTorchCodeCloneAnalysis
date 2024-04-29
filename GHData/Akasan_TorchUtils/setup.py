import setuptools


setuptools.setup(
    name="TorchUtils",
    version="1.0.0",
    install_requires=[],
    packages=[
        "TorchUtils",
        "TorchUtils.Analyzer",
        "TorchUtils.Core",
        "TorchUtils.DatasetGenerator",
        "TorchUtils.Layers",
        "TorchUtils.ModelGenerator",
        "TorchUtils.PipeLine",
        "TorchUtils.Trainer",
        "torchUtils.Visualizer"
    ]
)