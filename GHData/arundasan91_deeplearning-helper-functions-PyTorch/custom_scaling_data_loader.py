class CustomScaling(object):
    """Rotate image by a fixed angle which is ready for tranform.Compose()
    """

    def __init__(self, scale, angle=0, translate=[0,0], shear=0):
        self.scale = scale
        self.angle = angle
        self.translate = translate
        self.shear = shear

    def __call__(self, img):
        
        return transforms.ToTensor()(
            transforms.functional.affine(
                transforms.ToPILImage()(img), 
                self.angle, self.translate, self.scale, self.shear))
