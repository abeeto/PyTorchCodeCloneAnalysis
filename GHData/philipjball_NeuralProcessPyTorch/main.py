from neuralprocess import NeuralProcess, NeuralProcessLoss
from trainer import NeuralProcessTrainer
from utils import GPDataGenerator

def main():
    num_epochs = 100
    neural_process = NeuralProcess(1, 1, 10, 10, 10, width=200)
    data_generator = GPDataGenerator()
    trainer = NeuralProcessTrainer(neural_process, data_generator, num_epochs)
    trainer.train()

if __name__ == "__main__":
    main()
    