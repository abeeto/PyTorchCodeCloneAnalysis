# učitavanje najboljih težinskih vrijednosti dobivenih treniranjem na ImageNet skupu
weights = torchvision.models.AlexNet_Weights.DEFAULT 
weights

# direktoriji s trening i test podacima
#train_d = '/content/drive/MyDrive/data/grapevine/train' 
#test_d = '/content/drive/MyDrive/data/grapevine/test'

train_d = '/content/drive/MyDrive/data/traffic/train/train' 
test_d = '/content/drive/MyDrive/data/traffic/test/test'

#train_d = '/content/drive/MyDrive/data/Vegetable Images/train' 
#test_d = '/content/drive/MyDrive/data/Vegetable Images/test'

# učitavanje podataka u dataloader
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

train_data = datasets.ImageFolder(train_d, transform = preprocess)
test_data = datasets.ImageFolder(test_d, transform = preprocess)

class_names = train_data.classes

train_dataloader = DataLoader(
      train_data,
      batch_size=32,
      shuffle=True,
      num_workers=2,
      pin_memory=True,
  )
test_dataloader = DataLoader(
      test_data,
      batch_size = 32,
      shuffle = False,
      num_workers = 2,
      pin_memory = True,
  )

class_names

model = torchvision.models.alexnet(weights=weights).to(device)
summary(model=model, 
        input_size=(32, 3, 224, 224), # (batch_size, kanali, dimezije ulaznih slika)
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
)

# "zamrzavamo" sve trainable slojeve osim klasifikatora
for param in model.features.parameters():
    param.requires_grad = False

# ponovo ispisujemo sažetak modela da provjerimo ispravnost vrijednosti u "Trainable" stupcu
summary(model=model, 
        input_size=(32, 3, 224, 224), 
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
)    

torch.manual_seed(42)
torch.cuda.manual_seed(42)

# kreiramo novi klasifikator 
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.5), # vrijednost p prema Krizhevsky et al. 2012
    torch.nn.Linear(in_features=9216, # dimenzije izlaza prethodnog sloja
                    out_features=4096, 
                    bias=True),
    torch.nn.ReLU(),
    torch.nn.Dropout(p=0.5), # vrijednost p prema Krizhevsky et al. 2012
    torch.nn.Linear(in_features=4096, # dimenzije izlaza prethodnog sloja
                    out_features=4096, 
                    bias=True),
    torch.nn.ReLU(),
    torch.nn.Linear(in_features=4096, # dimenzije izlaza prethodnog sloja
                    out_features=len(class_names), # dimenzije izlaznog sloja = broj klasa u skupu podataka
                    bias=True)).to(device)
                    
# ponovo sažetak da provjerimo je li sve u redu
summary(model, 
        input_size=(32, 3, 224, 224), 
        verbose=0,
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
)

# standardni optimizer i loss function za AlexNet
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

# writer za train funkciju
writer = SummaryWriter("/content/drive/MyDrive/eksperimenti/AlexNet_grapevine")
model_path = '/content/drive/MyDrive/eksperimenti/AlexNet_grapevine/AlexGrabe_best_model.pt'
best_epoch_pred, best_epoch_true = [], []
# Start the timer
# from timeit import default_timer as timer 
# start_time = timer()


# fine-tuning modela na našem skupu podataka
results, best_epoch_true, best_epoch_pred = train(model = model,
                       train_dataloader = train_dataloader,
                       test_dataloader = test_dataloader,
                       optimizer = optimizer,
                       loss_fn = loss_fn,
                       epochs = 30,
                       es_patience = 3,
                       best_model = model_path,
                       labels=class_names,
                       device = device)


# End the timer and print out how long it took
#end_time = timer()
#print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")
