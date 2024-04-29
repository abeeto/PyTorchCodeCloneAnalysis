# code from https://stackoverflow.com/a/54428173

import numpy as np
import matplotlib.cm as cmaps
from matplotlib.colors import Normalize
import pyglet
import pyglet.gl

class ArrayImage:
    """Dynamic pyglet image of a 2d numpy array using matplotlib colormaps."""
    def __init__(self, array, cmap=cmaps.viridis, norm=None, rescale=True):
        self.array = array
        self.cmap = cmap
        if norm is None:
            norm = Normalize()
        self.norm = norm
        self.rescale = rescale

        self._array_normed = np.zeros(array.shape+(4,), dtype=np.uint8)
        # this line below was the bottleneck...
        # we have removed it by setting the _tex_data array to share the buffer
        # of the normalised data _array_normed
        # self._tex_data = (pyglet.gl.GLubyte * self._array_normed_data.size)( *self._array_normed_data )
        self._tex_data = (pyglet.gl.GLubyte * self._array_normed.size).from_buffer(self._array_normed)
        self._update_array()

        format_size = 4
        bytes_per_channel = 1
        self.pitch = array.shape[1] * format_size * bytes_per_channel
        self.image = pyglet.image.ImageData(array.shape[0], array.shape[1], "RGBA", self._tex_data)
        self._update_image()

    def set_array(self, data):
        self.array = data
        self.update()

    def _update_array(self):
        if self.rescale:
            self.norm.autoscale(self.array)
        self._array_normed[:] = self.cmap(self.norm(self.array), bytes=True)
        # don't need the below any more as _tex_data points to _array_normed memory
        # self._tex_data[:] = self._array_normed

    def _update_image(self):
        self.image.set_data("RGBA", self.pitch, self._tex_data)

    def update(self):
        self._update_array()
        self._update_image()
