import utils as U
import torch
import Math.vector3 as V3
import Math.quaternion as Q
from math import cos, sin, pi
import IK.api as api

def rosenbrock(x, y):
    a = torch.tensor([1.0])
    b = torch.tensor([100.0])
    return (a - x)**2 + b * (y - x**2)**2

def grad_rosenbrock(x, y):
    a = torch.tensor([1.0])
    b = torch.tensor([100.0])
    return torch.tensor([[-2 * (a - x) - 4 * b * (y - x**2) * x], [2 * b * (y - x**2)]])

def hess_rosenbrock(x, y):
    a = torch.tensor([1.0])
    b = torch.tensor([100.0])
    return torch.tensor([[2 - 4 * b * y + 12 * b * x**2, - 4 * b * x], [-4 * b * x, 2 * b]])


def test_grad():
    x0 = torch.tensor([-1.0, 1.0], requires_grad=True)
    fx = rosenbrock(*x0)

    auto_dx = U.jacobian(fx, x0)
    analytical_dx = grad_rosenbrock(*x0)

    assert(torch.norm(auto_dx - analytical_dx.T[0]) == 0)

    print("grad test success")

def test_hess():
    x0 = torch.tensor([-1.0, 1.0], requires_grad=True)
    fx = rosenbrock(*x0)

    auto_ddx = U.hessian(fx, x0)
    analytical_ddx = hess_rosenbrock(*x0)

    assert(torch.norm(analytical_ddx - auto_ddx) == 0)

    print("hess test success")

def test_vector3_zero():
    xex = torch.zeros((3,))
    x = V3.zero()

    assert(torch.norm(xex - x) == 0)

    print("vector 3 zero test success")

def test_vector3_make():
    xex = torch.tensor([1.0, 2.0, 3.0])
    x = V3.make(1, 2, 3)

    assert(torch.norm(xex - x) == 0)

    print("vector 3 make test success")

def test_vector3_i():
    xex = torch.tensor([1.0, 0.0, 0.0])
    x = V3.i()

    assert(torch.norm(xex - x) == 0)

    print("vector 3 i test success")

def test_vector3_j():
    xex = torch.tensor([0.0, 1.0, 0.0])
    x = V3.j()

    assert(torch.norm(xex - x) == 0)

    print("vector 3 j test success")

def test_vector3_k():
    xex = torch.tensor([0.0, 0.0, 1.0])
    x = V3.k()

    assert(torch.norm(xex - x) == 0)

    print("vector 3 k test success")

def test_vector3_cross():
    at = torch.tensor([1.0, 2.0, 3.0])
    bt = torch.tensor([3.0, 2.0, 1.0])
    ct = V3.cross(at, bt)

    import numpy as np
    anp = np.array([1.0, 2.0, 3.0])
    bnp = np.array([3.0, 2.0, 1.0])
    cnp = np.cross(anp, bnp)

    assert(torch.norm(ct - torch.from_numpy(cnp)) == 0)

    print("vector 3 cross test success")

def test_vector3_norm():
    at = torch.tensor([1.0, 2.0, 3.0])
    import numpy as np
    anp = np.array([1.0, 2.0, 3.0])

    assert(V3.norm(at) - np.linalg.norm(anp) == 0)

    print("vector 3 norm test success")


def test_quaternion_from_array():
    x = Q.from_array([1.0, 2.0, 3.0, 4.0])
    xex = torch.tensor([1.0, 2.0, 3.0, 4.0])

    assert(torch.norm(x - xex) == 0)

    print("quat from_array test success")

def test_quaternion_identity():
    i = Q.identity()
    iex = torch.tensor([1, 0, 0, 0], dtype=torch.float)

    assert(torch.norm(i - iex) == 0)

    print("quat identity test success")

def test_quaternion_Rx():
    radians = pi / 4.0
    rx = Q.Rx(radians)
    rxex = torch.tensor([cos(radians/2.0), sin(radians/2.0), 0.0, 0.0])

    assert(torch.norm(rx - rxex) == 0)

    print("quat Rx test success")

def test_quaternion_Ry():
    radians = pi / 4.0
    ry = Q.Ry(radians)
    ryex = torch.tensor([cos(radians/2.0), 0.0, sin(radians/2.0), 0.0])

    assert(torch.norm(ry - ryex) == 0)

    print("quat Ry test success")

def test_quaternion_Rz():
    radians = pi / 4.0
    rz = Q.Rz(radians)
    rzex = torch.tensor([cos(radians/2.0), 0.0, 0.0, sin(radians/2.0)])

    assert(torch.norm(rz - rzex) == 0)

    print("quat Rz test success")

def test_quaternion_conjugate():
    x = Q.conjugate(Q.from_array([1.0, 2.0, 3.0, 4.0]))
    xex = torch.tensor([1.0, -2.0, -3.0, -4.0])

    assert(torch.norm(x - xex) == 0)

    print("quat conjugate test success")

def test_quaternion_prod():
    qa = Q.from_array([1.0, 1.0, 0.0, 0.0])
    qb = Q.from_array([1.0, 0.0, 1.0, 0.0])

    p = Q.prod(qa, qb)

    pex = torch.tensor([1.0, 1.0, 1.0, 1.0])

    assert(torch.norm(p - pex) == 0)

    print("quat prod test success")

def test_quaternion_rotate():
    q = torch.tensor([1.0, 0.0, 1.0, 0.0])
    r = pi / 4.0
    r_ = torch.tensor([r, 0.0, 0.0])
    qr = Q.rotate(q, r_)

    # Numpy implementation
    import numpy as np
    def prod(qa, qb):
        a = qa[0]
        b = qb[0]
        A = qa[1:]
        B = qb[1:]
        qs = a * b - np.dot(A, B)
        qv = a * B + A * b + np.cross(A, B, axis=0)
        return np.array([qs, qv[0], qv[1], qv[2]])

    anp = np.array([1.0, 0.0, 1.0, 0.0])
    qrnp = np.array([0.0, r, 0.0, 0.0])
    qrex = prod(prod(anp, qrnp), np.array([anp[0], -anp[1], -anp[2], -anp[3]]))[1:]

    assert(torch.norm(torch.from_numpy(qrex) - qr) < 1e-4)

    print("quat rotate test success")

def test_IK_api_radians_to_degrees():
    import numpy as np
    assert(api.radians_to_degrees(0.14) == (180.0 * 0.14 / np.pi))

    print("IK.api radians to degrees test success")

if __name__ == '__main__':
    test_grad()
    test_hess()
    test_vector3_zero()
    test_vector3_make()
    test_vector3_i()
    test_vector3_j()
    test_vector3_k()
    test_vector3_cross()
    test_vector3_norm()
    test_quaternion_from_array()
    test_quaternion_identity()
    test_quaternion_Rx()
    test_quaternion_Ry()
    test_quaternion_Rz()
    test_quaternion_conjugate()
    test_quaternion_prod()
    test_quaternion_rotate()
    test_IK_api_radians_to_degrees()
