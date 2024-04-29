import numpy as np
import os
from torch.utils.data import DataLoader
import av

try:
    from Video.VideoReader import VideoReader
    from Video.VideoWriter import VideoWriter
    from Image.ImageSequenceWriter import ImageSequenceWriter
    from Image.ImageSequenceReader import ImageSequenceReader
    from Extensions.Extensions import Extensions
except:
    from .Video.VideoReader import VideoReader
    from .Video.VideoWriter import VideoWriter
    from .Image.ImageSequenceWriter import ImageSequenceWriter
    from .Image.ImageSequenceReader import ImageSequenceReader
    from .Extensions.Extensions import Extensions


class PyTorchMediaIO:
    def __init__(self, path=None, frame_rate=23.976, bit_rate=1000000, codec=None, pixel_format=None, profile=None):
        self.transform = None
        self.shape = None
        self.frame_shape = None
        self.Reader = None
        self.dir_path = os.path.dirname(__file__)
        self.stream = None
        self.extensions = None
        self.loaded = False
        self.profile = profile
        self.pixel_format = pixel_format
        self.codec = codec
        self.bit_rate = bit_rate
        self.frame_rate = frame_rate

        if path:
            self.path = path
            self.extensions = Extensions()
            self.mediaType = self.extensions.GetMediaType(path)
        else:
            self.path = None

    def GetMediaType(self):
        if self.extensions is None:
            self.extensions = Extensions()
        self.mediaType = self.extensions.GetMediaType(self.path)
        return self.mediaType

    def Open(self, path=None, transform=None):
        self.transform = transform
        if self.path:
            self.GetMediaType()
            if self.mediaType[0] == "Video":
                self.LoadVideo()
            elif self.mediaType[0] == "Raster Image":
                self.LoadImageSequence()

        elif path:
            self.path = path
            self.GetMediaType()
            if self.mediaType[0] == "Video":
                self.LoadVideo()
            elif self.mediaType[0] == "Raster image":

                self.LoadImageSequence()
            else:
                pass

    def LoadVideo(self):
        self.Reader = VideoReader(self.path, transform=self.transform)
        self.frame_rate = self.Reader.frame_rate
        self.shape = self.Reader.shape

    def LoadImageSequence(self):
        self.Reader = ImageSequenceReader(self.path, transform=self.transform)
        self.shape = self.Reader.shape
        # self.frame_shape = self.Reader.video.frame_shape

    def GetFrame(self, frameNumber):
        currentFrame = frameNumber - 1
        frame = None
        if 0 <= currentFrame < len(self.Reader):
            frame = self.Reader[currentFrame]
        else:
            frame = None
        return frame

    def GetFrameCount(self):
        return len(self.Reader)

    def GetFrameRate(self):
        if self.frame_rate:
            return self.frame_rate
        else:
            if self.mediaType[0] == "Video":
                self.Reader.frame_rate()
            else:
                return 1

    def GetDuration(self, frame_rate=None):
        if frame_rate is not None:
            self.frame_rate = frame_rate
        return self.frames_to_TC(self.GetFrameCount())

    def GetDataloaderFiles(self):
        if self.transform is None:
            return DataLoader(self.Reader.asNumpyArray())
        return DataLoader(self.Reader)

    def GetDimensions(self):
        if self.shape:
            return self.shape
        else:
            self.shape = self.Reader.getShape()
            return self.shape

    def SetOutput(self, path):
        self.path = path

    def Writer(self):
        if self.mediaType[0] == "Video":
            self.Writer = VideoWriter(path=self.path, frame_rate=self.frame_rate, bit_rate=self.bit_rate,
                                      codec=self.codec, pixel_format=self.pixel_format, profile=self.profile)
        else:
            self.Writer = ImageSequenceWriter(path=self.path, extension=self.mediaType[1].replace('.', ''))
        return self.Writer

    def frames_to_TC(self, frames):
        h = int(frames / 86400)
        m = int(frames / 1440) % 60
        s = int((frames % 1440) / self.frame_rate)
        f = frames % 1440 % self.frame_rate
        return "%02d:%02d:%02d:%02d" % (h, m, s, f)
