from setuptools import setup, find_packages

setup(
    name             = "torchsummaryM",
    version          = "1.0.0",
    description      = "Improved visualization tool of torchsummary & torchsummaryX.",
    author           = "Hongyeob Kim",
    author_email     = "fvl_mah@g.kmou.ac.kr",
    url              = "https://github.com/MaiHon/torchsummaryM",
    packages         =["torchsummaryM"],
    install_requires = ["torch", "numpy", "torchvision"],
)