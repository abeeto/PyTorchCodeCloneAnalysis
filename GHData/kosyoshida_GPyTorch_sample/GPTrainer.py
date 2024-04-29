class Trainer(object):

    def __init__(self, gpr, likelihood, optimizer, mll):
        self.gpr = gpr
        self.likelihood = likelihood
        self.optimizer = optimizer
        self.mll = mll

    def update_hyperparameter(self, epochs):
        self.gpr.train()
        self.likelihood.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            output = self.gpr(self.gpr.train_inputs[0])

            loss = - self.mll(output, self.gpr.train_targets)
            loss.backward()
            self.optimizer.step()

            if epoch % (epochs//10) == 0:
                print('Epoch %d/%d - Loss: %.3f ' % (epoch + 1, epochs, loss.item()))
