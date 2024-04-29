class CustomRotation(object):
    """Rotate image by a fixed angle which is ready for tranform.Compose()
    """

    def __init__(self, degrees, resample=False, expand=False, center=None):
        self.degrees = degrees
        self.resample = resample
        self.expand = expand
        self.center = center

    def __call__(self, img):
        
        return transforms.ToTensor()(
            transforms.functional.rotate(
                transforms.ToPILImage()(img), 
                self.degrees, self.resample, self.expand, self.center))
