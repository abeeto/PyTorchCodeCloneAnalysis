from setuptools import setup, find_packages

setup(
    name="pytorchablation",
    author="Alessio Molinari",
    author_email="alessio.molinari96@gmail.com",
    description="Ablation study framework for PyTorch",
    url="https://github.com/alessiomolinari/PyTorch_Ablation_Study",
    packages=find_packages(),
    install_requires=["torch", 'numpy'],
)
