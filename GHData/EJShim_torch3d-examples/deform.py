import os, sys
import torch
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.loss import chamfer_distance, mesh_edge_loss, mesh_laplacian_smoothing, mesh_normal_consistency
import numpy as np
import vtk
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


from pytorch3d.io import load_obj

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    print("CPU Mode")


#Debugging renderer
app = QApplication([])
iren = QVTKRenderWindowInteractor()
iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

renWin = iren.GetRenderWindow()
renWin.SetSize(1000, 1000)


ren = vtk.vtkRenderer()
renWin.AddRenderer(ren)



class Worker(QThread):
    backwarded = pyqtSignal(object)

    def __init__(self, src, target):
        super().__init__()
        self.src = src
        self.target = target




    def run(self):

        
        deform_verts = torch.full(self.src.verts_packed().shape, 0.0, device=device, requires_grad=True)
        optimizer = torch.optim.SGD([deform_verts], lr=1.0, momentum=0.9)


        Niter = 2000
        w_chamfer = 1.0
        w_edge = 1.0
        w_normal =  0.01
        w_laplacian = 0.1

        


        for i in range(Niter):
            
        
            optimizer.zero_grad()

            new_src_mesh = self.src.offset_verts(deform_verts)

            
            sampmle_trg = sample_points_from_meshes(self.target, 5000)
            sample_src = sample_points_from_meshes(new_src_mesh, 5000)


            loss_chamfer, _ = chamfer_distance(sampmle_trg, sample_src)
            loss_edge = mesh_edge_loss(new_src_mesh)
            loss_normal = mesh_normal_consistency(new_src_mesh)
            loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform")

            #weighted sum of the losses
            loss = loss_chamfer*w_chamfer + loss_edge*w_edge + loss_normal * w_normal + loss_laplacian * w_laplacian
            

            loss.backward()
            optimizer.step()
            print('total_loss = %.6f' % loss)
            self.backwarded.emit(new_src_mesh.verts_packed())


def convertMeshToPolyData(mesh):

    result = vtk.vtkPolyData()

    verts = mesh.verts_packed()
    faces = mesh.faces_packed()

    points = vtk.vtkPoints()
    for i in range(len(verts)):
        points.InsertNextPoint(verts[i][0], verts[i][1], verts[i][2])


    cells = vtk.vtkCellArray()    
    for i in range(len(faces)):
        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0, faces[i][0])
        triangle.GetPointIds().SetId(1, faces[i][1])
        triangle.GetPointIds().SetId(2, faces[i][2])
        cells.InsertNextCell(triangle)
    

    result.SetPoints(points)
    result.SetPolys(cells)

    return result


def convertTorchMesh(polydata):

    verts = []
    faces = []

    nPoints = polydata.GetNumberOfPoints()
    nCells = polydata.GetNumberOfCells()


    for i in range(nPoints):
        verts.append(polydata.GetPoint(i))
    verts = torch.tensor(verts, dtype=torch.float).to(device)
    for i in range(nCells):
        cell = polydata.GetCell(i)
        faces.append( [cell.GetPointId(0), cell.GetPointId(1), cell.GetPointId(2)] )
    faces = torch.tensor(faces, dtype=torch.long).to(device)

    
    #Normalize
    center = verts.mean(0)
    verts = verts - center
    scale = max(verts.abs().max(0)[0])
    verts = verts / scale

    #Convert
    torchMesh = Meshes(verts=[verts], faces=[faces])

    return torchMesh

def MakeActor(polydata):
    
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    return actor

def MakePointCloudActor(pointcloud):

    polydata = vtk.vtkPolyData()

    points = vtk.vtkPoints()
    for i in range(pointcloud.shape[0]):
        points.InsertNextPoint(pointcloud[i][0], pointcloud[i][1], pointcloud[i][2])

    polydata.SetPoints(points)

    mapper = vtk.vtkOpenGLSphereMapper()
    mapper.SetInputData(polydata)
    mapper.SetRadius(.005)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(1, 0, 0)

    return actor

def UpdatePolyData(polydata, tensor):

    nPoints = polydata.GetNumberOfPoints()

    for i in range(nPoints):
        polydata.GetPoints().SetPoint(i, [tensor[i][0], tensor[i][1], tensor[i][2]])
    polydata.GetPoints().Modified()

if __name__ == "__main__":
    reader = vtk.vtkOBJReader()
    reader.SetFileName("./dolphin.obj")
    reader.Update()

    polydata = reader.GetOutput()


    trg_mesh = convertTorchMesh(polydata)
    UpdatePolyData(polydata, trg_mesh.verts_packed())
    src_mesh = ico_sphere(4, device)
            


    srcPoly = convertMeshToPolyData(src_mesh)
    srcActor = MakeActor(srcPoly)
    srcActor.GetProperty().SetColor(0, 1, 0)
    trgActor = MakeActor(polydata)    

    ren.AddActor(srcActor)
    ren.AddActor(trgActor)
    renWin.Render()
    

    def onUpdate(data):
        UpdatePolyData(srcPoly, data)
        renWin.Render()
    trainingWorker = Worker(src_mesh, trg_mesh)
    trainingWorker.backwarded.connect(onUpdate)
    trainingWorker.start()

    window = QMainWindow()
    window.setCentralWidget(iren)
    window.show()


    sys.exit(app.exec_())

