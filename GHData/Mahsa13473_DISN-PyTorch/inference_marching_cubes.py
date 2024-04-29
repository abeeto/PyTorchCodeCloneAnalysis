import os

import argparse
import numpy as np

from skimage import measure


def write_off(file, vertices, faces, bminx, gridsizeX, bminy, gridsizeY, bminz, gridsizeZ ):
    """
    Writes the given vertices and faces to OFF.
    :param vertices: vertices as tuples of (x, y, z) coordinates
    :type vertices: [(float)]
    :param faces: faces as tuples of (num_vertices, vertex_id_1, vertex_id_2, ...)
    :type faces: [(int)]
    """

    num_vertices = len(vertices)
    num_faces = len(faces)

    assert num_vertices > 0
    assert num_faces > 0

    with open(file, 'w') as fp:
        fp.write('OFF\n')
        fp.write(str(num_vertices) + ' ' + str(num_faces) + ' 0\n')

        for vertex in vertices:
            assert len(vertex) == 3, 'invalid vertex with %d dimensions found (%s)' % (len(vertex), file)
            fp.write(str(vertex[0]*gridsizeX+bminx) + ' ' + str(vertex[1]*gridsizeY+bminy) + ' ' + str(vertex[2]*gridsizeZ+bminz) + '\n')

        for face in faces:
            assert len(face) == 3, 'only triangular faces supported (%s)' % (file)
            fp.write('3 ' + str(face[0]) + ' ' + str(face[1]) + ' ' + str(face[2]) + '\n')

        # add empty line to be sure
        fp.write('\n')




def marching_cubes(tensor):
    """
    Perform marching cubes using mcubes.
    :param tensor: input volume
    :type tensor: numpy.ndarray
    :return: vertices, faces
    :rtype: numpy.ndarray, numpy.ndarray
    """

    vertices, faces, normals, values = measure.marching_cubes_lewiner(tensor.transpose(2, 1, 0), 0) #(2, 1, 0)
    return vertices, faces


def infer(bmaxx, bmaxy, bmaxz, bminx, bminy, bminz, res, SDF, i):


    dimensionX = res
    dimensionY = res
    dimensionZ = res

    print(bminx, bminy, bminz)
    print(bmaxx, bmaxy, bmaxz)
    print(dimensionX)


    gridsizeX = (bmaxx - bminx)/dimensionX
    gridsizeY = (bmaxy - bminy)/dimensionY
    gridsizeZ = (bmaxz - bminz)/dimensionZ



    # SDF_grid = np.load('output/0_0.45009.npy') #global_all
    SDF_grid = SDF

    vertices, faces = marching_cubes(SDF_grid)

    a = res/64.0 # to solve different resolution problem

    off_file = '%s/%d.off' % ('final_output', i)
    write_off(off_file, vertices, faces, bminx, gridsizeX*a, bminy, gridsizeY*a, bminz, gridsizeZ*a)
    print('Wrote %s.' % off_file)

    print(bminx, gridsizeX, bminy, gridsizeY, bminz, gridsizeZ)
