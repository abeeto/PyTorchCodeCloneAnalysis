from visdom import Visdom
import numpy as np
import math
import os.path

viz = Visdom()

# textwindow = viz.text("Hello Pytorch")
#
# image_window = viz.image(
#     np.random.rand(3, 256, 256),
#     opts=dict(title="random", caption="random noise"))
#
# images_window = viz.images(
#     np.random.rand(10, 3, 64, 64),
#     opts=dict(title="random", caption="random noise"))

# Y = np.random.rand(100)
# scatter_window = viz.scatter(
#     X=np.random.rand(100, 2),
#     Y=(Y + 1.5).astype(int),
#     opts=dict(
#         legend=['Apples', 'Pears'],
#         xtickmin=0,
#         xtickmax=2,
#         xtickstep=0.5,
#         ytickmin=0,
#         ytickmax=2,
#         ytickstep=0.5,
#         markersymbol='cross-thin-open', ), )
#
# viz.updateTrace(
#     X=np.random.rand(50),
#     Y=np.random.rand(50),
#     win=scatter_window,
#     name='bananas', )
#
# viz.scatter(
#     X=np.random.rand(100, 3),
#     Y=(Y + 1.5).astype(int),
#     opts=dict(
#         legend=['Men', 'Women'],
#         markersize=5, ))
#
# viz.bar(X=np.random.rand(20))

# viz.bar(
#     X=np.abs(np.random.rand(5, 3)),
#     opts=dict(
#         stacked=True,
#         legend=['Facebook', 'Google', 'Twitter'],
#         rownames=['2012', '2013', '2014', '2015', '2016']))

# viz.bar(
#     X=np.random.rand(10, 3),
#     opts=dict(
#         stacked=False, legend=['The Netherlands', 'France', 'United States']))

# x = np.tile(np.arange(1, 101), (100, 1))
# y = x.transpose()
# X = np.exp((((x - 50)**2) + ((y - 50)**2)) / -(20.0**2))
#
# # contour
# viz.contour(X=X, opts=dict(colormap='Viridis'))
#
# # surface
# viz.surf(X=X, opts=dict(colormap='Hot'))

# viz.line(Y=np.random.rand(10))

# # line updates
# win = viz.line(
#     X=np.column_stack((np.arange(0, 10), np.arange(0, 10))),
#     Y=np.column_stack((np.linspace(5, 10, 10), np.linspace(5, 10, 10) + 5)), )
#
# viz.line(
#     X=np.column_stack((np.arange(10, 20), np.arange(10, 20))),
#     Y=np.column_stack((np.linspace(5, 10, 10), np.linspace(5, 10, 10) + 5)),
#     win=win,
#     update='append')

# pie chart
X = np.asarray([19, 26, 55])
viz.pie(X=X, opts=dict(legend=['Residential', 'Non-Residential', 'Utility']))

# PyTorch tensor
try:
    import torch
    viz.line(Y=torch.Tensor([[0., 0.], [1., 1.]]))
except ImportError:
    print('Skipped PyTorch example')
