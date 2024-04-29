import zipfile

import yadisk

# Get there: https://ramziv.com/article/8

y = yadisk.YaDisk(
    id="",
    secret="",
    token="",
)
y.check_token()
y.download(
    src_path="/gan-research/snow_dataset.zip",
    path_or_file="/home/ubuntu/cycle_gan_pytorch/snow_dataset.zip",
)

with zipfile.ZipFile(
    "/home/ubuntu/cycle_gan_pytorch/snow_dataset.zip", "r"
) as zip_ref:
    zip_ref.extractall("/home/ubuntu/cycle_gan_pytorch/data/")
