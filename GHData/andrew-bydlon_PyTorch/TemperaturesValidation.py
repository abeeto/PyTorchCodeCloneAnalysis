import torch
import torch.optim as optim


def model(t_u, w, b):
    return w * t_u + b


def loss_fn(t_p, t_c):
    squaredDiffs = (t_p - t_c)**2
    return squaredDiffs.mean()

t_c = torch.tensor([0.5,  14.0, 15.0, 28.0, 11.0,  8.0,  3.0, -4.0,  6.0, 13.0, 21.0])
t_u = torch.tensor([35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4])

params = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate = 1e-2
optimizer = optim.SGD([params], lr=learning_rate)

nSamples = t_u.shape[0]
nVal = int(0.2 * nSamples)

shuffledIndices = torch.randperm(nSamples)

trainIndices = shuffledIndices[:-nVal]
valIndices = shuffledIndices[-nVal:]

traint_u = t_u[trainIndices]
traint_c = t_c[trainIndices]

valt_u = t_u[valIndices]
valt_c = t_c[valIndices]

traint_un = traint_u/10
valt_un = valt_u/10


def trainingLoop(n_epochs, optimizer, params, traint_u, traint_c, valt_u, valt_c):
    for epoch in range(1, n_epochs + 1):
        traint_p = model(traint_u, *params)
        trainloss = loss_fn(traint_p, traint_c)

        with torch.no_grad():
            valt_p = model(valt_u, *params)
            valloss = loss_fn(valt_p, valt_c)
            assert valloss.requires_grad == False

        optimizer.zero_grad()
        trainloss.backward()
        optimizer.step()

        if epoch <= 3 or epoch % 500 == 0:
            print('Epoch {}, Training Loss {}, Validation loss {}'.format(epoch, float(trainloss), float(valloss)))
    return params

history = trainingLoop(n_epochs=3000, optimizer=optimizer, params=params, traint_u=traint_un, traint_c=traint_c, valt_u=valt_un, valt_c=valt_c)
