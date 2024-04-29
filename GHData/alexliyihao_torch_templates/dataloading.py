from torch.utils.data import Dataset, DataLoader

class abstract_Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]

img_list = None
transformations = transforms.ToTensor()
extracted_image_list = glob.glob("/content/KORUS/**/extracted_images", recursive= True)
# for all the extracted_image folder
for extracted_image in tqdm(extracted_image_list[:5], desc = "loading the image", leave = True):
    # for all the .tif subfolder
    for tif_name in tqdm(glob.iglob(os.path.join(extracted_image, "*.tif")), desc = extracted_image.split("/")[-2], leave = False):
        # get all the paths
        paths = glob.iglob(os.path.join(tif_name, "*.png"))
        # read in the images
        if img_list == None:
            img_list = torch.stack([transformations(Img.open(img).resize((64,64))) for img in paths])
        else:
            img_list = torch.cat((img_list, torch.stack([transformations(Img.open(img).resize((64,64))) for img in paths])), dim = 0)
        gc.collect()
phytoplankton_data = abstract_Dataset(imgs)
trainloader = DataLoader(phytoplankton_data, batch_size=BATCH_SIZE, shuffle=True)
