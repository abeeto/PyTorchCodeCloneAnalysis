from time import time

import torch
import torch_pruning as tp
from torchinfo import summary
from torchvision.models import resnet18

# model = resnet18(pretrained=True).eval()
# test_input = torch.randn(1, 3, 224, 224)
# large_test_input = torch.randn(32, 3, 224, 224)
from datasets.utils import ReduceBoundingBoxes
from models import ModelMeta
from models.PoolResnet import PoolResnet

num_of_patches = 10
input_shape = (480, 480)
test_input = torch.randn(1, 3, *input_shape)
large_test_input = torch.randn(10, 3, *input_shape)

model = PoolResnet(
    filters=128,
    input_shape=(3, *input_shape),
    num_of_patches=num_of_patches,
    num_of_residual_blocks=10,
)
model_setup = ModelMeta(model=model, lr=1e-4)
checkpoint = torch.load(
    "lightning_logs/custom_128_10x10_sam_adam/checkpoints/epoch=69-step=56279.ckpt",
    map_location=torch.device("cpu"),
)
model_setup.load_state_dict(checkpoint["state_dict"])
model.num_of_patches = num_of_patches
model.reduce_bounding_boxes = ReduceBoundingBoxes(
    probability_threshold=0.65,
    iou_threshold=0.1,
    input_shape=model.input_shape,
    num_of_patches=model.num_of_patches,
)
model = model.eval()

# 1. setup strategy (L1 Norm)
strategy = tp.strategy.L1Strategy()  # or tp.strategy.RandomStrategy()
# strategy = tp.strategy.RandomStrategy()

# 2. build layer dependency for resnet18
DG = tp.DependencyGraph()
DG.build_dependency(model, example_inputs=test_input)


def benchmark_model(model, input_tensor):
    model = model.cpu()
    s = time()
    for _ in range(10):
        model(input_tensor)
    print("Time Taken:", time() - s)


summary(model, test_input.shape)
benchmark_model(model, large_test_input)


def loop_over_layers(module):
    for layer in module.modules():
        if hasattr(layer, "weight"):
            pruning_fn = None
            if hasattr(layer, "is_pruned"):
                pass
            elif isinstance(layer, torch.nn.Conv2d):
                pruning_fn = tp.prune_conv
            elif isinstance(layer, torch.nn.BatchNorm2d):
                pruning_fn = tp.prune_batchnorm
            elif isinstance(layer, torch.nn.Linear):
                pruning_fn = tp.prune_linear

            if pruning_fn is not None:
                pruning_idxs = strategy(layer.weight, amount=0.2)
                pruning_plan = DG.get_pruning_plan(layer, pruning_fn, idxs=pruning_idxs)
                # print(pruning_plan)
                pruning_plan.exec()


loop_over_layers(model)

summary(model, test_input.shape)
benchmark_model(model, large_test_input)

i = 0
