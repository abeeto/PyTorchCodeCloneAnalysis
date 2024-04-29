from typing import Optional
import math

from torch import nn
import torch.nn.functional as F


module_defs = {
    # Table 1. and 2.
    # kernel, exp size, out channels, squeeze excite, nonlinearities, stride
    "large": [
        {"k": 3, "exp": 16, "oc": 16, "se": False, "act": "RE", "s": 1},
        {"k": 3, "exp": 64, "oc": 24, "se": False, "act": "RE", "s": 2},
        {"k": 3, "exp": 72, "oc": 24, "se": False, "act": "RE", "s": 1},
        {"k": 5, "exp": 72, "oc": 40, "se": True, "act": "RE", "s": 2},
        {"k": 5, "exp": 120, "oc": 40, "se": True, "act": "RE", "s": 1},
        {"k": 5, "exp": 120, "oc": 40, "se": True, "act": "RE", "s": 1},
        {"k": 3, "exp": 240, "oc": 80, "se": False, "act": "HS", "s": 2},
        {"k": 3, "exp": 200, "oc": 80, "se": False, "act": "HS", "s": 1},
        {"k": 3, "exp": 184, "oc": 80, "se": False, "act": "HS", "s": 1},
        {"k": 3, "exp": 184, "oc": 80, "se": False, "act": "HS", "s": 1},
        {"k": 3, "exp": 480, "oc": 112, "se": True, "act": "HS", "s": 1},
        {"k": 3, "exp": 672, "oc": 112, "se": True, "act": "HS", "s": 1},
        {"k": 5, "exp": 672, "oc": 160, "se": True, "act": "HS", "s": 2},
        {"k": 5, "exp": 960, "oc": 160, "se": True, "act": "HS", "s": 1},
        {"k": 5, "exp": 960, "oc": 160, "se": True, "act": "HS", "s": 1},
    ],
    "small": [
        {"k": 3, "exp": 16, "oc": 16, "se": True, "act": "RE", "s": 2},
        {"k": 3, "exp": 72, "oc": 24, "se": False, "act": "RE", "s": 2},
        {"k": 3, "exp": 88, "oc": 24, "se": False, "act": "RE", "s": 1},
        {"k": 5, "exp": 96, "oc": 40, "se": True, "act": "HS", "s": 2},
        {"k": 5, "exp": 240, "oc": 40, "se": True, "act": "HS", "s": 1},
        {"k": 5, "exp": 240, "oc": 240, "se": True, "act": "HS", "s": 1},
        {"k": 5, "exp": 120, "oc": 48, "se": True, "act": "HS", "s": 1},
        {"k": 5, "exp": 144, "oc": 48, "se": True, "act": "HS", "s": 1},
        {"k": 5, "exp": 288, "oc": 96, "se": True, "act": "HS", "s": 2},
        {"k": 5, "exp": 576, "oc": 96, "se": True, "act": "HS", "s": 1},
        {"k": 5, "exp": 576, "oc": 96, "se": True, "act": "HS", "s": 1},
    ],
}
initialisations = {
    "normal": nn.init.normal_,
    "xavier": nn.init.xavier_normal_,
    "kaiming": nn.init.kaiming_normal_,
}


def calc_pad(k: int) -> int:
    return (k - 1) // 2


class HardSwish(nn.Module):  # 5.2 Nonlinearities
    def __init__(self, inplace=False):
        super(HardSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3.0, inplace=self.inplace) * 1.0 / 6.0 * x


class HardSigmoid(nn.Module):  # 5.2 Nonlinearsities
    def __init__(self, inplace=False):
        super(HardSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3.0, inplace=self.inplace) * 1.0 / 6.0


class SqueezeExcite(nn.Module):
    def __init__(
        self, c, r=4, inplace=True
    ):  # 5.3 Large squeeze-and-excite, reduce to 1/4
        super(SqueezeExcite, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(
            1
        )  # ((N), C, output_size, output_size), reduces to ((N), C, 1, 1)
        self.excite = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace),
            nn.Linear(c // r, c, bias=False),
            HardSigmoid(inplace),
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excite(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)


class InvertedBottleNeck(nn.Module):
    def __init__(
        self,
        ic: int,
        oc: int,
        k: int,
        exp: int,
        s: int,
        se: bool,
        act: str,
        drop_rate: Optional[float] = None,
    ):
        super(InvertedBottleNeck, self).__init__()
        self.pw1 = nn.Sequential(nn.Conv2d(ic, exp, 1, bias=False), nn.BatchNorm2d(exp))
        self.dw = nn.Sequential(
            nn.Conv2d(
                exp, exp, k, stride=s, groups=exp, padding=calc_pad(k), bias=False
            ),
            nn.BatchNorm2d(exp),
        )
        self.pw2 = nn.Sequential(nn.Conv2d(exp, oc, 1, bias=False), nn.BatchNorm2d(oc))
        self.se = SqueezeExcite(exp) if se else None
        self.drop_out = nn.Dropout(drop_rate) if drop_rate else None
        self.residual_connection = True if ic == oc and s == 1 else False

        if act == "RE":
            self.act = nn.ReLU(inplace=True)
        elif act == "HS":
            self.act = HardSwish(inplace=True)
        else:
            raise NotImplementedError

    def forward(self, x):
        residual = x
        y = self.pw1(x)
        y = self.act(y)
        y = self.dw(y)
        y = self.act(y)

        if self.se:
            y = self.se(y)

        y = self.pw2(y)

        if self.drop_out:
            y = self.drop_out(y)

        if self.residual_connection:
            return y + residual
        else:
            return y


class Classifier(nn.Module):
    def __init__(self, head_type: str, last_in: int, last_out: int, n_classes: int):
        super(Classifier, self).__init__()
        self.head_type = head_type
        if self.head_type == "fc":
            self.classification_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Linear(last_out, n_classes), nn.Softmax(1)
            )
        elif self.head_type == "conv":
            self.classification_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(last_in, last_out, 1, stride=1),
                HardSwish(inplace=True),
                nn.Conv2d(last_out, n_classes, 1),
            )
        else:
            raise NotImplementedError

    def forward(self, x):
        x = x.mean(3).mean(2) if self.head_type == "fc" else x
        y = self.classification_head(x)

        if self.head_type == "conv":
            return y.view(y.shape[0], -1)
        else:
            return y


class MobileNetV3(nn.Module):  # TODO
    def __init__(
        self,
        model_size: str,
        n_classes: int,
        head_type: str,
        initialisation: Optional[str] = "normal",
        drop_rate: Optional[float] = None,
        alpha=1.0,
    ):
        assert (
            model_size == "small" or model_size == "large"
        ), "model_size should be 'small' or 'large'."
        assert 0.0 < alpha <= 1.0, "Width multiplier (alpha) is a value (0, 1]"

        super(MobileNetV3, self).__init__()

        if model_size == "small":
            last_in = self._divisible(576 * alpha) if alpha < 1.0 else 576
            last_out = self._divisible(1024 * alpha) if alpha < 1.0 else 1024
        elif model_size == "large":
            last_in = self._divisible(960 * alpha) if alpha < 1.0 else 960
            last_out = self._divisible(1280 * alpha) if alpha < 1.0 else 1280

        self.model = self._build_model(model_size, last_in, drop_rate, alpha)
        self.classifier = Classifier(head_type, last_in, last_out, n_classes)
        self._init_weights(initialisation)

    def forward(self, x):
        y = self.model(x)
        y = self.classifier(y)
        return y

    def _build_model(
        self, model_size: str, last_in: int, drop_rate: float, alpha: float
    ) -> nn.Sequential:
        modules = nn.Sequential()
        ic = self._divisible(16 * alpha) if alpha < 1.0 else 16

        # Build the first block.
        block_0 = nn.Sequential(
            nn.Conv2d(3, ic, 3, stride=2, padding=calc_pad(3), bias=False),
            nn.BatchNorm2d(ic),
            HardSwish(inplace=True),
        )
        modules.append(block_0)

        # Build the MobileNet bottleneck blocks.
        defs = module_defs[model_size]
        for bn in defs:
            oc = self._divisible(bn["oc"] * alpha) if alpha < 1.0 else bn["oc"]
            exp = self._divisible(bn["exp"] * alpha) if alpha < 1.0 else bn["exp"]
            modules.append(
                InvertedBottleNeck(
                    ic, oc, bn["k"], exp, bn["s"], bn["se"], bn["act"], drop_rate
                )
            )
            ic = oc

        # Build the last few blocks.
        modules.append(nn.Conv2d(ic, last_in, 1, stride=1, bias=False))
        modules.append(nn.BatchNorm2d(last_in))
        modules.append(HardSwish(inplace=True))

        return modules

    def _init_weights(self, initialisation):
        init_function = initialisations[initialisation]
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init_function(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _divisible(self, n, divisible_by=4):
        return int(math.ceil(n * 1.0 / divisible_by) * divisible_by)


if __name__ == "__main__":
    import torch

    torch.manual_seed(42)
    model = MobileNetV3(
        model_size="small",
        n_classes=10,
        head_type="conv",
        initialisation="kaiming",
        drop_rate=0.8,
        alpha=0.8,
    )
    x = torch.randn(16, 3, 32, 32)
    results = model(x)
    print(results.argmax(1))
