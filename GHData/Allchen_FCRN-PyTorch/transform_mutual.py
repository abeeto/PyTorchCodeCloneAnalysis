import numpy.random as random


class TransformMutual:
    def __init__(self, img_size, output_size, rescale_range,
                 flip_rate=0.5, rotate_degree=5
                 ):
        rand = random.rand()
        if rand < flip_rate:
            flip = True
        else:
            flip = False
        rescale = random.rand() * (rescale_range[1]
                                   - rescale_range[0]) + rescale_range[0]
        rotate = ((random.rand() * 2 - 1) * rotate_degree + 360) % 360
        translation_x1 = int(img_size[0] * (rescale - 1) * random.rand())
        translation_y1 = int(img_size[1] * (rescale - 1) * random.rand())
        translation_x2 = int(translation_x1 + img_size[0])
        translation_y2 = int(translation_y1 + img_size[1])

        self.flip = flip
        self.rescale = rescale
        self.rotate = rotate
        self.translation_x1 = translation_x1
        self.translation_x2 = translation_x2
        self.translation_y1 = translation_y1
        self.translation_y2 = translation_y2
        self.output_size = output_size
        return
