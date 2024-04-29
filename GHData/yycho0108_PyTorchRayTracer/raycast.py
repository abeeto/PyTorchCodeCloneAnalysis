#!/usr/bin/env python3

import torch as tr


def dot(a: tr.Tensor, b: tr.Tensor):
    return tr.einsum('...i,...i->...', a, b)


def cross(a: tr.Tensor, b: tr.Tensor):
    """ ([...]3 x (...)3 -> [...](...)3) """
    # Convert to compatible shapes.
    a_ = a.view(-1, 1, 3)
    b_ = b.view(1, -1, 3)
    a_, b_ = tr.broadcast_tensors(a_, b_)
    # Compute cross product.
    out = tr.cross(a_, b_, dim=-1)
    # Convert back to format.
    out = out.view(a.shape[:-1] + b.shape[:-1] + (3,))
    return out


def ray_triangle_intersection(
        ray_origin: tr.Tensor, ray_vector: tr.Tensor,
        triangle: tr.Tensor,
        broadcast_triangle: bool = True,
        max_distance=100.0):
    if broadcast_triangle:
        shape = ray_origin.shape[:-1] + triangle.shape[:-2]
        triangle = triangle.view(1, -1, 3, 3)
    else:
        # triangle already given as (R,T,3,3)
        shape = triangle.shape[:-2]
        triangle = triangle.view((ray_origin.numel()//3,) + (-1, 3, 3))

    # Broadcast ray per each triangle.
    ray_origin = ray_origin.view(-1, 1, 3)
    ray_vector = ray_vector.view(-1, 1, 3)

    kEpsilon = 0.0000001
    v0, v1, v2 = [triangle[..., i, :] for i in range(3)]
    e10 = v1 - v0
    e20 = v2 - v0
    ray_vector, e20 = tr.broadcast_tensors(ray_vector, e20)
    h = tr.cross(ray_vector, e20, dim=-1)  # RT3
    e10, h = tr.broadcast_tensors(e10, h)
    a = dot(e10, h)
    f = 1.0 / a
    s = ray_origin - v0
    u = f * dot(s, h)
    q = s.cross(e10)
    v = f * dot(ray_vector, q)
    t = f * dot(e20, q)

    miss = (((a > -kEpsilon) & (a < kEpsilon)) |
            ((u < 0.0) | (u > 1.0)) |
            ((v < 0.0) | (u+v > 1.0)) |
            ((t < kEpsilon) | (t > max_distance)))

    point = ray_origin + ray_vector * t[..., None]

    # Reshape to format
    miss = miss.view(shape)
    t = t.view(shape)
    point = point.view(shape + (3,))
    hit = tr.logical_not(miss)

    return (hit, t)


def get_tangent(u: tr.Tensor):
    c1 = cross(u, tr.tensor([1.0, 0.0, 0.0]))
    c2 = cross(u, tr.tensor([0.0, 1.0, 0.0]))
    tangent = tr.where(c1.norm(dim=-1, keepdim=True) >
                       c2.norm(dim=-1, keepdim=True), c1, c2)
    return tangent.view(u.shape)


def test_ray_triangle_intersection():
    kNumRays = 16
    kNumTriangles = 1
    kMaxRayCastDistance = 100.0

    # Create unit norm ray.
    ray_origin = tr.randn(kNumRays, 3)
    ray_dir = tr.randn(kNumRays, 3)
    ray_dir.div_(ray_dir.norm(dim=-1, keepdim=True))

    # Compute orthogonal basis.
    tangent = get_tangent(ray_dir)
    tangent.div_(tangent.norm(dim=-1, keepdim=True))
    bitangent = tangent.cross(ray_dir)
    bitangent.div_(bitangent.norm(dim=-1, keepdim=True))

    # Create triangle.
    # triangle = tr.randn(kNumTriangles, 3, 3)
    sel_idx = tr.randint(0, kNumRays, (kNumTriangles,))
    tri_dirs = ray_dir[sel_idx]
    tri_dist = 10.0 * tr.rand(kNumTriangles)
    tri_t1 = tangent[sel_idx]
    tri_t2 = bitangent[sel_idx]
    tri_ctr = ray_origin[sel_idx] + \
        (tri_dirs * tri_dist[..., None])
    v0 = tri_ctr + 1.0 * tri_t1
    v1 = tri_ctr - 0.5 * tri_t1 - 0.5 * tri_t2
    v2 = tri_ctr - 0.5 * tri_t1 + 0.5 * tri_t2
    triangle = tr.stack([v0, v1, v2], dim=-2)  # (T,3,3)

    hits, dists = (ray_triangle_intersection(ray_origin,
                                             kMaxRayCastDistance*ray_dir, triangle))


def main():
    test_ray_triangle_intersection()


if __name__ == '__main__':
    main()
