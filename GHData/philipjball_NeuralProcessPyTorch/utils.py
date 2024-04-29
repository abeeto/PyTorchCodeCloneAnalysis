import numpy as np
from sklearn import gaussian_process
from sklearn.model_selection import train_test_split

class GPDataGenerator():
    
    def __init__(self, num_funcs: int=30, seed: int=1234):
        self.num_funcs = num_funcs
        self.f = gaussian_process.GaussianProcessRegressor()
        self.x_grid = np.linspace(-3, 3, 100)
        self.y_grid = self.f.sample_y(self.x_grid.reshape(-1, 1), self.num_funcs, seed)
    
    def return_y_eval(self, x_eval: np.array, y_index: int):
        return np.interp(x_eval, self.x_grid, self.y_grid[:, y_index]).reshape(-1)
    
    def create_training_set(self, num_points: int=30, test_split: float=0.2):
        x_context = []
        y_context = []
        x_target = []
        y_target = []
        for i in range(self.num_funcs):
            x = np.sort(np.random.uniform(-3, 3, num_points))
            y = self.return_y_eval(x, i)
            x_c, _, y_c, _ = train_test_split(x, y, test_size=test_split)
            x_c_sort = np.argsort(x_c)
            x_context.append(x_c[x_c_sort])
            y_context.append(y_c[x_c_sort])
            x_target.append(x)
            y_target.append(y)
        return x_context, x_target, y_context, y_target