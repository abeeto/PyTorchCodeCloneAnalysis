# Göğüs kanseri için oluşturulmuş veri seti kullanılarak (30 özelliğe sahip veri seti ile)
# Hücrenin özelliklerine göre iyi huylu mu kötü huylu mu olup olmadığının tespit edilmesi
import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer

device = torch.device("cpu")

# Hyper Parameter

input_size = 30  # Girdilerin sayısı (Toplam 30 kolon)
hidden_size = 500  # Ara katman 500 sinir hücresi olsun
num_classes = 2  # Kaç adet sınıfım olduğu (2 adet - 1 kötü huylu, 2 iyi huylu)
num_epoch = 10000  # Tekrar sayısı

learning_rate = 1e-5

girdi, cikti = load_breast_cancer(return_X_y=True)  # Numpy array olarak girdi ve çıktıları aldık

# Numpy arraylerini PyTorch arraylerine çevirmek lazım
train_input = torch.from_numpy(girdi).float()  # float?
train_output = torch.from_numpy(cikti).long()  # float?


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Veri setimizdeki 30 özellikten 500 node a çıkardık.
        self.lrelu = nn.LeakyReLU(negative_slope=0.02)  # Aktivasyon Fonksiyonu
        self.fc2 = nn.Linear(hidden_size, num_classes)  # 500 node dan 2 node indirdik.

    def forward(self, inputt):
        outfc1 = self.fc1(inputt)  # ilk katman girdi
        outfc1lrelu = self.lrelu(outfc1)  # ardından lrelu ya girip çıktı olarak değer döndü
        out = self.fc2(outfc1lrelu)  # 2 çıktıdan birinin tahminini veren yapı
        return out


model = NeuralNet(input_size, hidden_size, num_classes)

lossf = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epoch):
    outputs = model(train_input)
    loss = lossf(outputs, train_output)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epoch, loss.item()))
