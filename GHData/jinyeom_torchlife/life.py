import numpy as np
import torch as pt
from torch.distributions import Bernoulli
from torch.nn import functional as F
from pyglet import gl
from pyglet.window import Window
from pyglet.image import ImageData
from torchvision.utils import make_grid

device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')

class ImageViewer:
  def __init__(self):
    self.window = None
    self.isopen = False
    self.display = None

  def __del__(self):
    self.close()

  def imshow(self, arr, caption):
    height, width, _ = arr.shape
    if self.window is None:
      self.width = width
      self.height = height
      self.isopen = True
      self.window = Window(width=width, 
                           height=height, 
                           display=self.display, 
                           vsync=False, 
                           resizable=True)
    assert len(arr.shape) == 3
    image = ImageData(width, height, 'RGB', arr.tobytes(), pitch=-3*width)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, 
                       gl.GL_TEXTURE_MAG_FILTER, 
                       gl.GL_NEAREST)
    texture = image.get_texture()
    texture.width = self.width
    texture.height = self.height
    self.window.clear()
    self.window.switch_to()
    self.window.dispatch_events()
    texture.blit(0, 0)
    self.window.flip()
    self.window.set_caption(caption)

  def close(self):
    if self.isopen:
      self.window.close()
      self.isopen = False

class Life:
  def __init__(self, size, n=1):
    self.grid = pt.zeros(n, 1, size, size).to(device)
    self.mask = pt.tensor([[[
      [1, 1, 1], 
      [1, 0, 1], 
      [1, 1, 1]
    ]]], dtype=pt.float).to(device)
    self.viewer = ImageViewer()
    self.t = 0

  def reset(self):
    self.grid.fill_(0)
    self.t = 0

  def rand_init(self, seed=0, p=0.2):
    pt.manual_seed(seed)
    self.grid = Bernoulli(p).sample(self.grid.size()).to(device)

  def render(self, invert=False):
    b, c, h, w = self.grid.shape
    nrow = int(np.ceil(np.sqrt(b)))
    grid = make_grid(1-self.grid, nrow=nrow)
    grid = grid if invert else 1-grid
    grid = 255 * grid.to(pt.uint8).cpu()
    grid = grid.numpy().transpose((1, 2, 0))
    self.viewer.imshow(grid, caption=f't={self.t}')

  def step(self):
    padded = F.pad(self.grid, (1, 1, 1, 1), mode='circular')
    neighbors = F.conv2d(padded, self.mask)
    mask0 = (neighbors < 2).to(pt.float) * self.grid
    mask1 = (neighbors > 3).to(pt.float) * self.grid
    mask2 = (neighbors == 3).to(pt.float) * (1 - self.grid)
    self.grid[mask0.to(pt.bool)] = 0
    self.grid[mask1.to(pt.bool)] = 0
    self.grid[mask2.to(pt.bool)] = 1
    self.t += 1

  def close(self):
    self.viewer.close()

if __name__ == '__main__':
  game = Life(100, n=64)
  game.rand_init()
  while True:
    game.render()
    game.step()
  
