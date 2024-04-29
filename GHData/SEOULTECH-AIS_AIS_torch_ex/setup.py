from setuptools import setup

setup(
    name="torch_ex",
    version="1.0.0",
    description="Custom base code module for pytorch",
    url="https://github.com/SEOULTECH-AIS-gyounghun6612/AIS_torch_ex.git",
    author="Choi_keonghun & Jun_eins",
    author_email="dev.gyounghun6612@gmail.com",
    packages=["torch_ex"],
    zip_safe=False,
    install_requires=[
        "torch", "python_ex"
    ]
)
