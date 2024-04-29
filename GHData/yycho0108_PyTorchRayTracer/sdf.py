import torch as tr


def sdf_plane(point: tr.Tensor):
    """ Distance to plane whose normal is x-axis """
    return point[..., 0]


def sdf_sphere(point: tr.Tensor, radius: tr.Tensor):
    """ Distance to sphere """
    return tr.norm(point, dim=-1) - radius


def sdf_box(point: tr.Tensor, box: tr.Tensor):
    """ Distance to box """
    d = tr.abs(point) - box
    lhs = tr.relu(tr.max(d, dim=-1).values)
    rhs = tr.norm(tr.relu(d), dim=-1)
    return lhs + rhs


def sdf_cylinder(point: tr.Tensor, radius: tr.Tensor, height: tr.Tensor):
    distance = tr.norm(point[..., :2], dim=-1) - radius
    distance = tr.max(distance, tr.abs(point[..., 2]) - height)
    return distance


def sdf_torus(point: tr.Tensor, r_major: tr.Tensor, r_minor: tr.Tensor):
    """ Torus """
    a = tr.norm(point[..., :2], dim=-1) - r_major
    return tr.norm([a, point[..., 2]], dim=0) - r_minor

def main():
    with tr.cuda.device(0):
        print(tr.cuda.current_device())
        dist = sdf_plane(tr.randn(3))
        print(dist)
        dist = sdf_sphere(tr.randn(3), tr.randn(()))
        print(dist)
        dist = sdf_box(tr.randn(3), tr.randn(3))
        print(dist)
        dist = sdf_cylinder(tr.randn(3), tr.randn(()), tr.randn(()))
        print(dist)

if __name__ == '__main__':
    main()
