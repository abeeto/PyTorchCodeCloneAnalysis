"""
This script loads an image, extracts sprites, and infer the classes with the network trained earlier.
"""
from src.models import LeNet
from src.detection import get_box_contours, get_sprites, preprocess_sprites
import imageio
import torch
import collections
import matplotlib.pyplot as plt

LABELS=[1, 2]

Box = collections.namedtuple('Box', 'contour sprite label')

def process(image, model, debug=None):
    """
    This function processes an image given a model, and returns a list of Box.
    """

    debug_inner = debug in ["inner", "all"]
    contours = get_box_contours(image, debug=debug_inner)
    sprites = get_sprites(image, contours, debug=debug_inner)
    inputs = preprocess_sprites(sprites, debug=debug_inner)
    labels = [model.infer(i) for i in inputs]

    boxes = [Box(contour=c, sprite=s, label=l) for c, s, l in  zip(contours, sprites, labels)]

    if debug in ["all", "synthesis"]:
        for box in boxes:
            fig, ax = plt.subplots(nrows=2)
            ax[0].imshow(image)
            ax[0].plot(box.contour[:,0], box.contour[:,1], "og")
            ax[0].plot(box.contour.mean(axis=0)[0], box.contour.mean(axis=0)[1], "og")
            ax[1].imshow(box.sprite)
            ax[1].set_title("Label recognized: {}".format(box.label))
            plt.show()

    return boxes


if __name__ == "__main__":

    import glob

    labels = [1, 2]

    checkpoint_path = "checkpoints/final-23:45:12.t7"

    model = LeNet(classes=labels)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    test = glob.glob('data/cubes/**/*.jpg', recursive=True)

    for path in test:
        print("Testing image {}".format(path))
        image = imageio.imread(path)[:,:,0]
        try:
            boxes = process(image, model, debug="synthesis")
        except Exception:
            print("Failed to process image {}".format(path))
            pass
