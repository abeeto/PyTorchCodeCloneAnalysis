from OcelotMinimal.cpbd import elements
from OcelotMinimal.cpbd import optics
from OcelotMinimal.cpbd.magnetic_lattice import MagneticLattice


class Lattice(MagneticLattice):
    """Abstract lattice class."""
    def __init__(self, elementList):
        # purpose of this?
        method = optics.MethodTM()
        method.global_method = optics.SecondTM


        super().__init__(elementList, method=method)

        # log element positions
        self.positions = list()
        self.endPositions = list()
        self.logElementPositions()

        return

    def getTransferMaps(self, dim=4):
        for i, tm in enumerate(optics.get_map(self, self.totalLen, optics.Navigator(self))):
            # get type and length of element
            element = self.sequence[i]
            elementLength = self.sequence[i].l

            # get transfer matrices
            rMatrix = tm.r_z_no_tilt(tm.length, 0)  # first order 6x6
            tMatrix = tm.t_mat_z_e(tm.length, 0)  # second order 6x6x6

            # cutoff longitudinal coordinates
            rMatrix = rMatrix[:dim, :dim]
            tMatrix = tMatrix[:dim, :dim, :dim].reshape((dim, -1))

            yield rMatrix, tMatrix, element, elementLength

    def logElementPositions(self):
        """Store beginning and end of each element."""
        self.positions = list()
        self.endPositions = list()
        totalLength = 0

        for element in self.sequence:
            self.positions.append(totalLength)
            totalLength += element.l
            self.endPositions.append(totalLength)



class DummyLattice(Lattice):
    """Playground."""
    def __init__(self, angle: float, e1: float, e2: float):
        # specify beam line elements
        d1 = elements.Drift(1)
        rb1 = elements.RBend(0.1, angle=angle, e1=e1, e2=e2)
        d2 = elements.Drift(1)
        
        # set up beam line
        cell = [d1, rb1, d2,]

        super().__init__(cell)

        return


class SIS18_Cell_minimal(Lattice):
    """SIS18 cell consisting of dipoles and quadrupoles."""
    def __init__(self, k1f: float = 3.12391e-01, k1d: float = -4.78047e-01):
        # default focusing strengths correspond to hor. / vert. tune of 4.2 resp. 3.3

        # specify beam line elements
        rb1 = elements.RBend(l=2.617993878, angle=0.2617993878, e1=0.1274090354, e2=0.1274090354)
        rb2 = elements.RBend(l=2.617993878, angle=0.2617993878, e1=0.1274090354, e2=0.1274090354)

        qs1f = elements.Quadrupole(l=1.04, k1=k1f)
        qs2d = elements.Quadrupole(l=1.04, k1=k1d)
        qs3t = elements.Quadrupole(l=0.4804, k1=2 * k1f)

        d1 = elements.Drift(0.645)
        d2 = elements.Drift(0.9700000000000002)
        d3 = elements.Drift(6.839011704000001)
        d4 = elements.Drift(0.5999999999999979)
        d5 = elements.Drift(0.7097999999999978)
        d6 = elements.Drift(0.49979999100000283)

        # set up beam line
        cell = [d1, rb1, d2, rb2, d3, qs1f, d4, qs2d, d5, qs3t, d6]

        super().__init__(cell)

        return


class SIS18_Cell_sext(Lattice):
    """SIS18 cell consisting of dipoles and quadrupoles."""
    def __init__(self):
        # specify beam line elements
        rb1 = elements.RBend(l=2.617993878, angle=0.2617993878, e1=0.1274090354, e2=0.1274090354)
        rb2 = elements.RBend(l=2.617993878, angle=0.2617993878, e1=0.1274090354, e2=0.1274090354)

        k1f = 3.12391e-01   # tune: 4.2 (whole ring)
        k1d = -4.78047e-01  # tune: 3.3
        qs1f = elements.Quadrupole(l=1.04, k1=k1f)
        qs2d = elements.Quadrupole(l=1.04, k1=k1d)
        qs3t = elements.Quadrupole(l=0.4804, k1=2 * k1f)

        ks1c = elements.Sextupole(l=0.32, k2=0)
        ks3c = elements.Sextupole(l=0.32, k2=0)

        d1 = elements.Drift(0.645)
        d2 = elements.Drift(0.9700000000000002)
        d3a = elements.Drift(6.345)
        d3b = elements.Drift(0.175)
        d4 = elements.Drift(0.5999999999999979)
        d5a = elements.Drift(0.195)
        d5b = elements.Drift(0.195)
        d6 = elements.Drift(0.49979999100000283)

        # set up beam line
        cell = [d1, rb1, d2, rb2, d3a, ks1c, d3b, qs1f, d4, qs2d, d5a, ks3c, d5b, qs3t, d6]

        super().__init__(cell)

        return


class SIS18_Cell(Lattice):
    """SIS18 cell consisting of dipoles and quadrupoles."""
    def __init__(self, k1f: float = 3.12391e-01, k1d: float = -4.78047e-01):
        # specify beam line elements
        rb1a = elements.RBend(l=2.617993878 / 2, angle=0.2617993878 / 2, e1=0.1274090354, e2=0)
        rb1b = elements.RBend(l=2.617993878 / 2, angle=0.2617993878 / 2, e1=0, e2=0.1274090354)
        rb2a = elements.RBend(l=2.617993878 / 2, angle=0.2617993878 / 2, e1=0.1274090354, e2=0)
        rb2b = elements.RBend(l=2.617993878 / 2, angle=0.2617993878 / 2, e1=0, e2=0.1274090354)

        qs1f = elements.Quadrupole(l=1.04, k1=k1f)
        qs2d = elements.Quadrupole(l=1.04, k1=k1d)
        qs3t = elements.Quadrupole(l=0.4804, k1=2 * k1f)

        ks1c = elements.Sextupole(l=0.32, k2=0)
        ks3c = elements.Sextupole(l=0.32, k2=0)

        hKick1 = elements.Hcor(0)
        hKick2 = elements.Hcor(0)
        vKick = elements.Vcor(0)

        hMon = elements.Monitor(0.13275)
        vMon = elements.Monitor(0.13275)

        d1 = elements.Drift(0.2)
        d2 = elements.Drift(0.9700000000000002)
        d3a = elements.Drift(6.345)
        d3b = elements.Drift(0.175)
        d4 = elements.Drift(0.5999999999999979)
        d5a = elements.Drift(0.195)
        d5b = elements.Drift(0.195)
        d6 = elements.Drift(0.49979999100000283)
        d6a = elements.Drift(0.3485)
        d6b = elements.Drift(0.3308)

        # set up beam line
        cell = [d1, rb1a, hKick1, rb1b, d2, rb2a, hKick2, rb2b, d3a, ks1c, d3b, qs1f, vKick, d4, qs2d, d5a, ks3c, d5b, qs3t, d6a, hMon, vMon, d6b]

        super().__init__(cell)

        return


class SIS18_Lattice(Lattice):
    """SIS18 multiturn lattice consisting of dipoles and quadrupoles."""
    def __init__(self, k1f: float = 3.12391e-01, k1d: float = -4.78047e-01, nPasses: int = 1):
        # set up cell
        cell = SIS18_Cell(k1f, k1d)

        # ring consists of 12 cells
        lattice = list()
        for i in range(12 * nPasses):
            lattice += cell.sequence

        super(SIS18_Lattice, self).__init__(lattice)
        return



class SIS18_Lattice_minimal(Lattice):
    """SIS18 multiturn lattice consisting of dipoles and quadrupoles."""
    def __init__(self, k1f: float = 3.12391e-01, k1d: float = -4.78047e-01, nPasses: int = 1):
        # set up cell
        cell = SIS18_Cell_minimal(k1f, k1d)

        # ring consists of 12 cells
        lattice = list()
        for i in range(12 * nPasses):
            lattice += cell.sequence

        super(SIS18_Lattice_minimal, self).__init__(lattice)
        return


class SIS18_Cell_hCor(Lattice):
    """SIS18 cell with two horizontal correctors."""
    def __init__(self):
        # specify beam line elements
        rb1 = elements.RBend(l=2.617993878, angle=0.2617993878, e1=0.1274090354, e2=0.1274090354)
        rb2 = elements.RBend(l=2.617993878, angle=0.2617993878, e1=0.1274090354, e2=0.1274090354)

        k1f = 3.12391e-01   # tune: 4.2 (whole ring)
        k1d = -4.78047e-01  # tune: 3.3
        qs1f = elements.Quadrupole(l=1.04, k1=k1f)
        qs2d = elements.Quadrupole(l=1.04, k1=k1d)
        qs3t = elements.Quadrupole(l=0.4804, k1=2 * k1f)

        d1 = elements.Drift(0.645)
        d2 = elements.Drift(0.9700000000000002)
        d3a = elements.Drift(2)
        d3b = elements.Drift(1)
        d3c = elements.Drift(2.839011704000001)
        d4 = elements.Drift(0.5999999999999979)
        d5 = elements.Drift(0.7097999999999978)
        d6 = elements.Drift(0.49979999100000283)

        hCor1 = elements.Hcor(0.5)
        hCor2 = elements.Hcor(0.5)

        # set up beam line
        cell = [d1, rb1, d2, rb2, d3a, hCor1, d3b, hCor2, d3c, qs1f, d4, qs2d, d5, qs3t, d6]

        super().__init__(cell)

        return



if __name__ == "__main__":
    lattice = SIS18_Cell()

    for r, t, element, length in lattice.getTransferMaps(dim=6):
        print("{} length={}".format(type(element), length))

        if type(element) is elements.Vcor:
            # print(r, t)
            pass

    lastQuad = lattice.sequence[-2]

