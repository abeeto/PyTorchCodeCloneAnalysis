import numpy

import torch
import torch.nn as nn

from OcelotMinimal.cpbd import elements

from TorchOcelot.Elements import LinearMap, SecondOrderMap


class Model(nn.Module):
    def __init__(self, lattice):
        super(Model, self).__init__()
        self.lattice = lattice
        return

    def forward(self, x, nTurns: int = 1, outputPerElement: bool = False, outputAtBPM: bool = False):
        if outputPerElement:
            outputs = list()
            for turn in range(nTurns):
                for m in self.maps:
                    x = m(x)
                    outputs.append(x)

            return torch.stack(outputs).permute(1, 2, 0)  # particle, dim, element
        elif outputAtBPM:
            outputs = list()
            for turn in range(nTurns):
                for m in self.maps:
                    x = m(x)

                    if type(m.element) is elements.Monitor:
                        outputs.append(x)

            return torch.stack(outputs).permute(1, 2, 0)  # particle, dim, element
        else:
            for turn in range(nTurns):
                for m in self.maps:
                    x = m(x)

            return x

    def setTrainable(self, category: str):
        """Enable training for matching elements."""
        self.requires_grad_(False)

        if category == "all":
            self.requires_grad_(True)
        elif category == "quadrupoles":
            for map in self.maps:
                if type(map.element) is elements.Quadrupole:
                    map.requires_grad_(True)
        elif category == "correctors":
            for map in self.maps:
                if type(map.element) in [elements.Hcor, elements.Vcor]:
                    map.requires_grad_(True)
        elif category == "magnets":
            for map in self.maps:
                if type(map.element) in [elements.RBend, elements.Quadrupole, elements.Sextupole]:
                    map.requires_grad_(True)

        return


class LinearModel(Model):
    def __init__(self, lattice, dim=4, dtype: torch.dtype = torch.float32):
        super().__init__(lattice)
        self.dim = dim
        self.dtype = dtype

        # create maps
        self.maps = nn.ModuleList()

        for rMatrix, _, element, _ in lattice.getTransferMaps(dim=dim):
            layer = LinearMap(element, rMatrix, dtype=dtype)
            self.maps.append(layer)

        return

    def symplecticRegularization(self, quadsOnly=False):
        """Sum up symplectic regularization penalties from all layers."""
        penalties = list()

        if quadsOnly:
            for map in self.maps:
                if type(map.element) is elements.Quadrupole:
                    penalties.append(map.symplecticRegularization())
        else:
            for map in self.maps:
                penalties.append(map.symplecticRegularization())

        penalties = torch.stack(penalties)
        return penalties.sum()

    def getTunes(self) -> list:
        oneTurnMap = self.oneTurnMap()

        xTrace = oneTurnMap[:2, :2].trace()
        xTune = torch.acos(1 / 2 * xTrace).item() / (2 * numpy.pi)

        if self.dim == 4 or self.dim == 6:
            yTrace = oneTurnMap[2:4, 2:4].trace()
            yTune = torch.acos(1 / 2 * yTrace).item() / (2 * numpy.pi)

            return [xTune, yTune]

        return [xTune, ]

    def oneTurnMap(self):
        # calculate one-turn map
        oneTurnMap = torch.eye(self.dim, dtype=self.dtype)
        for m in self.maps:
            oneTurnMap = torch.matmul(m.weight, oneTurnMap)

        return oneTurnMap


class SecondOrderModel(Model):
    def __init__(self, lattice, dim=4, dtype: torch.dtype = torch.float32):
        super().__init__(lattice)
        self.dim = dim

        # create maps
        self.maps = nn.ModuleList()

        for rMatrix, tMatrix, element, _ in lattice.getTransferMaps(dim=dim):
            layer = SecondOrderMap(element, rMatrix, tMatrix, dtype=dtype)
            self.maps.append(layer)

        return

    def firstOrderOneTurnMap(self):
        # calculate one-turn map
        oneTurnMap = torch.eye(self.dim)
        for m in self.maps:
            oneTurnMap = torch.matmul(m.w1.weight, oneTurnMap)

        return oneTurnMap



if __name__ == "__main__":
    from TorchOcelot.Lattice import SIS18_Cell

    # create model
    dim = 6
    dtype = torch.float32
    lattice = SIS18_Cell()
    model = LinearModel(lattice, dim=dim, dtype=dtype)

    # symplectic reg. buggy?
    for m in model.maps:
        print(type(m.element), m.symplecticRegularization())

        if type(m.element) is elements.RBend:
            print(m.weight)
