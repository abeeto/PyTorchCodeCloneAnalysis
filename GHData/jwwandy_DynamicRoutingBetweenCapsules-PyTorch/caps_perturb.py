import argparse
import models
import torch
import torchvision
import datasets
from torch.utils import data

class Inferrer:
    def __init__(self, model_path, use_cuda=True, patch_size=(1, 28, 28), route_iters=3):
        state = torch.load(model_path)
        self.model = models.CapsNet(patch_size=patch_size,
                                     route_iters=route_iters,
                                     with_reconstruction=True)
        self.model.load_state_dict(state['model_state'])
        if use_cuda and torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                self.model = torch.nn.DataParallel(self.model)
            else:
                self.model = self.model.cuda()
        self.model.eval()

    def infer_reconstruct(self, x, target):
        _, pred, rc = self.model(x, target)
        return pred, rc

    def infer_capsules(self, x):
        caps, _, _ = self.model(x)
        return caps

    def infer(self, x):
        _, pred, _ = self.model(x)
        return pred

    def reconstruct(self, caps):
        return self.model.reconstruct(caps)


def caps_dim_perturb(rc_cap):

    rc_cap = rc_cap.unsqueeze(1).expand(-1, 11, -1)
    perturb = torch.zeros_like(rc_cap)
    for i, sample in enumerate(perturb):
        sample[:, i] = torch.arange(-0.25, 0.3, 0.05).to(rc_cap)
    return rc_cap.add(perturb).reshape(-1, 16)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Model state file path")
    parser.add_argument("--output", type=str, default="rc_perturb.png", help="output image file path")
    parser.add_argument("--route-iter", type=int, default=3, help="routing iterations of capsules model")
    parser.add_argument("--use-cuda", type=bool, default=True)

    args = parser.parse_args()

    use_cuda = args.use_cuda and torch.cuda.is_available()
    inferrer = Inferrer(args.model_path, use_cuda)

    _, mnist_test = datasets.get_shift_MNIST()
    test_loader = data.DataLoader(mnist_test, batch_size=16, shuffle=True,
                                  num_workers=4, pin_memory=True)
    img, target = next(iter(test_loader))

    if args.use_cuda:
        img, target = img.cuda(), target.cuda()

    with torch.no_grad():
        caps = inferrer.infer_capsules(img)
        rc_cap = caps[torch.arange(16, dtype=torch.long), target.argmax(dim=1), :]
        rc_cap_perturb = caps_dim_perturb(rc_cap)
        rc_image = inferrer.reconstruct(rc_cap_perturb)
        torchvision.utils.save_image(rc_image, args.output, nrow=11)

