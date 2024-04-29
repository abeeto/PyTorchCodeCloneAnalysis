import csv
import os
from pathlib import Path
import shutil
from typing import List, Tuple
from urllib.error import URLError
from torchvision.datasets import ImageFolder,VisionDataset
from torchvision.datasets.utils import check_integrity, extract_archive, verify_str_arg,download_and_extract_archive
import logging
import subprocess



def check_exists(root:Path,resources:List[Tuple[str,str]]) -> bool:
        return all(check_integrity(root/file,md5) for file, md5 in resources)

class EyePACS(VisionDataset):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    default_folder = Path("~").expanduser()/".torchvision/"/"eyepacs/"
    main_zip_name ="diabetic-retinopathy-detection.zip"
    main_zip_hash = "PUT_HASH_HERE"

    resources = [
        ("diabetic-retinopathy-detection.zip", "PUT_HASH_HERE"),
    ]

    def __init__(self, root: Path="", split: str = "train",transform=None, target_transform=None) -> None:
        if isinstance(root,str):
            root = Path(root)
        # assert split in ["train","val","test"]
        root = root.expanduser()
        images_root = root
        super().__init__(images_root,transform=transform,target_transform=target_transform)
        

        self.download()
        self.preprocess()
    
    def preprocess(self,):
        main_zip_path = self.root/EyePACS.main_zip_name
        print(f"Extracting {main_zip_path} to {self.root}...")
        extract_archive(main_zip_path,self.root)

    def download(self,):
        main_zip_resource = [(EyePACS.main_zip_name,EyePACS.main_zip_hash)]
        if check_exists(self.root,main_zip_resource):
            return
        command = "kaggle competitions download -c diabetic-retinopathy-detection"
        current_path = Path.cwd()
        os.chdir(self.root)
        print("Downloading dataset (88.29gbs) this may take a while...")
        subprocess.run(command)
        os.chdir(current_path)
        if not check_exists(self.root,main_zip_resource):
            raise OSError(f"File {EyePACS.main_zip_name} has not been downloaded correctly.")
        




if __name__ == '__main__':
    from torchvision import transforms

    logging.basicConfig(level=logging.INFO)
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(EyePACS.mean,EyePACS.std)]
        )

    split ="val"
    dataset = EyePACS(Path("~/.torchvision/eyepacs/"),split=split,transform=transform)
    n = len(dataset)
    print(f"Eyepacs, split {split}, has  {n} samples.")
    # print("Showing some samples")
    # for i in range(0,n,n//5):
    #     image,klass = dataset[i]
    #     print(f"Sample of class {klass:3d}, image shape {image.shape}, words {dataset.idx_to_words[klass]}")


