import os
from torchvision.transforms import ToTensor

from PyTorchMediaIO import PyTorchMediaIO


def TestMediaLoader():
    testMediaMp4 = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                "testMedia/Anime/test1/test1_.mp4")
    testMediaMov = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                "testMedia/Anime/test1/test1_.mov")
    testMediaImg = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                "testMedia/Anime/test1/img/test1_004.png")
    testMediaOutpr422 = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                     "testMedia/Anime/test1/Out/test1_Prores422.mov")
    testMediaOutprpng = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                     "testMedia/Anime/test1/Out/img/01/test1_.png")
    testMediaOutprEXR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                     "testMedia/Anime/test1/Out/img/exr/test1_.exr")
    mediaLoader = PyTorchMediaIO()
    # mediaLoader.Open(path=testMediaMov)
    mediaLoader.Open(path=testMediaMov, transform=ToTensor())
    size = mediaLoader.GetDimensions()
    FrameCount = mediaLoader.GetFrameCount()
    MediaType = mediaLoader.GetMediaType()
    DataloaderFiles = mediaLoader.GetDataloaderFiles()
    Duration = mediaLoader.GetDuration()
    # Duration30 = mediaLoader.GetDuration(30)

    # ExportProresVideo(testMediaOutpr422,mediaLoader.frame_rate, DataloaderFiles)
    ExportImageSequence(testMediaOutprEXR, DataloaderFiles)


def ExportProresVideo(OutputPath,FrameRate,DataloaderFiles):
    # Export Prores Video
    mediaExporter = PyTorchMediaIO(path=OutputPath, frame_rate=FrameRate, bit_rate=50000,
                                   codec='prores', pixel_format='yuv422p10le', profile=3)

    mediaExporterWriter = mediaExporter.Writer()
    for src in DataloaderFiles:
        mediaExporterWriter.write(src)
    mediaExporterWriter.close()

def ExportImageSequence(OutputPath,DataloaderFiles):
    # Export Prores Video
    mediaExporter = PyTorchMediaIO(path=OutputPath)

    mediaExporterWriter = mediaExporter.Writer()
    count = 0
    for src in DataloaderFiles:
        print("Processing: " + str(count) + "/" + str(len(DataloaderFiles)))
        mediaExporterWriter.write(src)
        count += 1
    print("Finished Processing: " + str(len(DataloaderFiles)) + " Files")

    mediaExporterWriter.close()


if __name__ == '__main__':
    TestMediaLoader()
