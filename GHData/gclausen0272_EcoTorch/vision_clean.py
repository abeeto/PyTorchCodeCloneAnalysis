
model = models.resnet18(pretrained=True)

num_classes = 3

# Freezing the weights
for param in model.parameters():
    param.required_grad = False


# Replacing the final layer
model.fc =  nn.Sequential(nn.Linear(512, 256), 
                         nn.ReLU(), 
                         nn.Dropout(p=0.5), 
                         nn.Linear(256, num_classes), 
                         nn.LogSoftmax(dim=1))
