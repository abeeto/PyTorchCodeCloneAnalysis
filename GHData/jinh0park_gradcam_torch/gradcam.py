import torch
import torch.nn as nn
import torch.nn.functional as F
from model import ZeroNet
import numpy as np
from PIL import Image
from data_utils import STL10Loader
import matplotlib.pyplot as plt
from tempfile import TemporaryFile


def main(num_result=(3,3)):
    '''
    This function applies the gradcam and makes the saliency maps of test input imagesself.
    The results are saved to 'gradcam_result/' by .png format.

    Inputs:
    - num_result: A tuple containing the numbers of result image -> (correct, incorrect)

    Outputs:
    None

    '''
    model = ZeroNet(num_classes=10)
    PATH = 'saved_model'
    model.load_state_dict(torch.load(PATH, map_location='cpu'))

    stl10 = STL10Loader()

    sampler = iter(stl10.get_test_sampler())
    model.eval()
    correct = []
    incorrect = []
    while True:
        X, y = next(sampler)
        a, b = model(X).argmax(), y
        if a.eq(b.view_as(a)).item():
            if len(correct) < num_result[0]:
                correct.append((X, y))
        else:
            if len(incorrect) < num_result[1]:
                incorrect.append((X, y))
        if len(correct) == num_result[0] and len(incorrect) == num_result[1]:
            break
    for flag, result_set in [('correct', correct), ('incorrect', incorrect)]:

        for i, item in enumerate(result_set):
            model.zero_grad()
            X_sample, y_sample = item
            features = model.features(X_sample)
            features.requires_grad_()
            features_shape = features.size()
            features = features.view(features.size(0), -1)
            scores = model.classifier(features)
            s = scores.squeeze()[y]
            features.retain_grad()
            s.backward()

            A = features.grad.view(*features_shape).squeeze()
            alpha = A.sum(dim=(1,2)) / (A.size(1) * A.size(2))
            alpha, A = alpha.detach().numpy(), A.detach().numpy()
            L_gradcam = alpha.reshape(-1,1,1) * A
            L_gradcam = L_gradcam.sum(axis=0)
            L_gradcam = np.maximum(L_gradcam, 0)

            with TemporaryFile() as img_origin, TemporaryFile() as img_gradcam:
                # [Reference] How to get rid of everything beside the image:
                # https://stackoverflow.com/a/26610602/10086634

                fig = plt.figure()
                fig.patch.set_visible(False)

                ax = fig.add_subplot(111)

                plt.axis('off')
                plt.imshow(L_gradcam, interpolation='bicubic', cmap='nipy_spectral')

                extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                plt.savefig(img_gradcam, bbox_inches=extent)

                X_img = X_sample.squeeze().numpy()
                plt.imsave(img_origin, X_img.transpose((1,2,0)))

                img_origin_pil = Image.open(img_origin)
                img_gradcam_pil = Image.open(img_gradcam)

                img_origin_pil = img_origin_pil.resize((192, 192))
                img_gradcam_pil = img_gradcam_pil.resize((192, 192))

                img_result = Image.blend(img_origin_pil, img_gradcam_pil, 0.5)

                img_origin_pil.save('./gradcam_result/{}-origin-{}.png'.format(flag, i))
                img_result.save('./gradcam_result/{}-saliency-{}.png'.format(flag, i))


if __name__ == "__main__":
    main()
