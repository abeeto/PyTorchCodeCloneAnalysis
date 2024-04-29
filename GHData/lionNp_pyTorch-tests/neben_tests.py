import torch
import math
import daten_sample_und_funktion

sample_size = daten_sample_und_funktion.sample_size

#x = torch.linspace(-math.pi, math.pi, sample_size)
#x = torch.linspace(-2, 2, sample_size)
#x = torch.tensor([0,1,2,3])*1.0
#y = torch.sin(x)

x = torch.tensor(daten_sample_und_funktion.input)*1.0
y = torch.tensor(daten_sample_und_funktion.output)*1.0


power = torch.tensor([1, 2, 3])

# unsqueeze macht aus einem 1d array der länge n ein nd array der länge 1
# pow erweitert jede zeile des arrays indem es den ursprünglichen eintrag hoch das jeweilige element des tensors nimmt. die länge der zeile ist dann gleich der
# länge des tensors
xx = x.unsqueeze(-1).pow(power)

# sequentielle bearbeitung der schritte
# outputs werden mittels linearem Modell von den Inputs erzeugt
# flatten quetscht putput in 1d tensor (gleich der Form des outputs y)
model = torch.nn.Sequential(
    torch.nn.Linear(3, 1),
    torch.nn.Flatten(0, 1)
)

# lossfunktion mean squared errors
loss_fn = torch.nn.MSELoss(reduction='sum')

#lernrate
learning_rate = 1e-6

for t in range(sample_size):

    # abschätzung für y errechnen, davon loss ableiten
    y_pred = model(xx)
    loss = loss_fn(y_pred, y)

    # vor backpropagation gradient auf null setzen
    model.zero_grad()
    loss.backward()

    # gewichte mittels gradientenabstieg anpassen
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad

# zugriff auf layer wie auf item aus liste
linear_layer = model[0]

# For linear layer, its parameters are stored as `weight` and `bias`.
print(f'Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:, 0].item()} x + {linear_layer.weight[:, 1].item()} x^2 + {linear_layer.weight[:, 2].item()} x^3')
