import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def accuracy(loader, model, arch="NN"):
    if loader.dataset.train:
        print("checking on train")
    else:
        print("checking on test")

    correct_count = 0
    total = 0
    model.eval()  # to change working mode of model

    with torch.no_grad():  # to turn off gradients computation
        for data, targets in loader:

            if arch == "RNN":
                data = data.squeeze(1)  # batchx1x28x28 to batchx28x28

            data = data.to(device)
            targets = targets.to(device)

            if arch == "NN":
                data = data.reshape(data.shape[0], -1)

            score = model(data)
            _, preds = score.max(dim=1)
            correct_count += (preds == targets).sum()
            total += preds.shape[0]

        print(f"{correct_count}/{total} - accuracy {float(correct_count) / float(total) * 100:.3f}")
    model.train()
