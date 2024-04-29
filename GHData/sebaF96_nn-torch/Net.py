import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot
import time

"""
torchvision nos provee una coleccion de datos para fines educativos. En la vida real
vamos a tener que recolectar, preparar, formatear, etc. nuestro dataset y probablemente
ésta sea la parte mas tediosa y larga de construir y entrenar redes neuronales.

Para este ejemplo vamos a usar torchvision.datasets.MNIST (Digitos del 0 al 9 dibujados a mano)
"""

# Definimos info de entrenamiento e info para test. Transformamos las imagenes a tensor (torch.FloatTensor)
training_data = datasets.MNIST('', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
testing_data = datasets.MNIST('', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

# Creamos datasets con la informacion descargada y transformada, otra vez uno para entrenamiento y otro para test
trainset = DataLoader(training_data, batch_size=8, shuffle=True)    # Batch size & shuffle
testset = DataLoader(testing_data, batch_size=8, shuffle=True)      # GENERALIZATION (Evitar que la red "tome atajos")


class Net(nn.Module):   # Creamos nuestra red como una clase normal de Python que hereda de nn.Module

    def __init__(self):
        # Inicializar nn.Module
        super().__init__()

        """
        definimos nuestras "fully connected" layers igualandolas a nn.Linear(input, output).
        para terminar de entender esto, se piensa a las layers como "en el medio" de input/output para cada
        llamada a nn.Linear(i, o)
        """
        self.fc1 = nn.Linear(28 * 28, 64)   # 28x28 = tamaño de input layer. 64 = tamaño de la siguiente hidden layer.
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)    # Output layer = 10 neuronas (1 para cada digito)


    def forward(self, data):
        # Metodo que define como la data "va a fluir" por nuestra red

        """
        Una de las ventajas de PyTorch es la libertad para personalizar esta parte.
        En este caso pasamos la informacion por todas las layers en el orden que las definimos
        (no necesariamente tiene q ser asi) y para cada uno llamamos a torch.nn.functional.relu
        (rectified linear unit function element-wise) que sería nuestra Funcion de activacion,
        la cual corre en el lado de 'output' de nuestras self.fcs
        """
        data = F.relu(self.fc1(data))
        data = F.relu(self.fc2(data))
        data = F.relu(self.fc3(data))
        data = self.fc4(data)

        return F.log_softmax(data, dim=1)



if __name__ == '__main__':
    net = Net()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    """
    parameters() son los parametros "ajustables" de nuestra red. Por ahora todos los pesos y biases.
    learning rate (lr) puede verse como el tamaño de los pasos que dara nuestra red mientras esta aprendiendo,
    no debe ser ni muy grande ni muy chico ya que esto haria que se quede atascada. El lr se puede ir modificando
    mientras avanza el entrenamiento (es comun ir achicandolo a medida que la red va entrenando).
    Para este ejemplo quedará siempre asi.
    """

    # Funcion que itera sobre el trainset y ajusta pesos y biases
    def train():
        # Iteraremos 3 veces sobre nuestro trainset completo
        EPOCHS = 3

        for epoch in range(EPOCHS):
            for batch in trainset:  # data es cada lote o batch, en nuestro caso lotes de 8 imagenes y 8 labels
                images, labels = batch
                net.zero_grad()

                # Pasamos el batch por la red
                output = net(images.view(-1, 28 * 28))

                # Calculamos el loss
                loss = F.nll_loss(output, labels)
                loss.backward()

                # Ajustamos el peso
                optimizer.step()

            # La red comienza a entrenar, y para cada EPOCH vamos viendo el loss
            print(loss)

    # Funcion que itera sobre el testset y al final muestra el porcentaje de imagenes que predijo correctamente
    def evaluate():
        with torch.no_grad():
            net.eval()  # No queremos que nuestra red tome en cuenta estos resultados ya que es fase de evaluacion

            corrects = total = 0

            for batch in testset:
                images, labels = batch

                output = net(images.view(-1, 28 * 28))

                for idx, i in enumerate(output):
                    if torch.argmax(i) == labels[idx]:
                        corrects += 1
                    total += 1

        print(f"Nuestra red predijo correctamente un {round(corrects / total * 100, 2)}% de las imagenes")

    # Funcion que nos muestra 8 imagenes y las predicciones de la red.
    def slow_evaluation():
        with torch.no_grad():
            net.eval()

            # Agarramos el primer batch del lote de prueba
            for batch in testset:
                # Iteramos sobre las 8 imagenes
                for img in batch[0]:

                    # Metemos la imagen a la red y almacenamos la salida
                    output = net(img.view(-1, 28 * 28))

                    # Mostramos la prediccion de la red
                    print(f"Creo que esta imagen es un {torch.argmax(output)}, vos que decis?")

                    # Esperamos un ratito y mostramos la imagen
                    time.sleep(1.5)
                    pyplot.imshow(img.view(28, 28))
                    pyplot.show()
                break




    train()
    # evaluate()
    slow_evaluation()
