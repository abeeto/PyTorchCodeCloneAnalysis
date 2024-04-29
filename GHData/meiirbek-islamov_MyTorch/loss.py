import numpy as np

class MSELoss:

    def forward(self, A, Y):
        self.A = A
        self.Y = Y
        N = A.shape[0]
        C = A.shape[1]
        SE = (A - Y) * (A - Y)
        SSE = np.dot(np.dot(np.ones((1, N), dtype="f"), SE), np.ones((C, 1), dtype="f"))
        L = SSE/(N * C)
        return L

    def backward(self):
        dLdA = self.A - self.Y
        return dLdA

class CrossEntropyLoss:

    def forward(self, A, Y):
        self.A = A
        self.Y = Y
        N = A.shape[0]
        C = A.shape[1]
        self.softmax = np.exp(A)/((np.exp(A) @ np.ones((C, 1), dtype="f")) @ np.ones((1, C), dtype="f"))
        crossentropy = -Y * np.log(self.softmax)
        sum_crossentropy = (np.ones((1, N), dtype="f") @ crossentropy) @ np.ones((C, 1), dtype="f")
        loss = sum_crossentropy/N
        return loss


    def backward(self):
        dLdA = self.softmax - self.Y
        return dLdA
