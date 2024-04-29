from lib.Data import data_loader
from lib.Trainer import TrainerBase
from models.AutoEncoder import AutoEncoder

if __name__ == "__main__":
	t = TrainerBase(data_loader=data_loader, model=AutoEncoder(), epochs=10)
	t.run()
