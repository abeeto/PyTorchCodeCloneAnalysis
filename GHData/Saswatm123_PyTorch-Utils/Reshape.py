class Reshape(nn.Module):
    def __init__(self, *target_shape):
        super().__init__()
        self.target_shape = target_shape
    
    def forward(self, x):
        return x.view(*self.target_shape)