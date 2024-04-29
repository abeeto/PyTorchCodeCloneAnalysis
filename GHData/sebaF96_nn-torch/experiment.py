from Net import trainset, Net
from matplotlib import pyplot
import torch


def print_dataset():
    # Imprimimos el primer batch del trainset (8 imagenes, cada una con su etiqueta)
    for data in trainset:
        print(data)
        break

    # img_tensor es el tensor de la primer imagen del primer batch, label es su respectiva etiqueta
    input()
    img_tensor, label = data[0][0], data[1][0]
    print(img_tensor)
    print(label)

    # mostramos la imagen
    input()
    pyplot.imshow(img_tensor.view(28, 28))
    pyplot.show()


def show_balance():
    # Que tan balanceado esta nuestro dataset?
    total = 0
    counter = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}

    for data in trainset:
        imgs, labels = data

        for lab in labels:
            counter[int(lab)] += 1
            total += 1

    for label, quantity in counter.items():
        print(f"El numero {label} aparece {quantity} veces ({str(quantity/total * 100)[:4]} %)")


def data_trough_network():
    net = Net()
    # print(net)

    # Generamos un tensor aleatorio del mismo tamaño que nuestras imagenes, y se lo damos como input a la red
    random_image = torch.rand(28, 28).view(-1, 28 * 28)

    # Almacenamos la salida de la red e imprimimos el resultado (predicciones) por ahora sin sentido
    output = net(random_image)
    print(output)


if __name__ == '__main__':
    #  print_dataset()
    #  show_balance()
    data_trough_network()

    pass
