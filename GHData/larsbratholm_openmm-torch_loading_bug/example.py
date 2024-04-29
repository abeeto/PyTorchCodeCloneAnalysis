import numpy as np
import openmmtorch
import torch
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *

class ParallelModel(torch.nn.Module):
    """
    Simple model where the energy is a sum of two linear
    transformations of the positions.
    Each linear model is on a different device.
    """

    def __init__(self, device0, device1):
        super().__init__()
        self.device0 = device0
        self.device1 = device1
        self.l0 = torch.nn.Linear(3 * 23, 1).to(self.device0)
        self.l1 = torch.nn.Linear(3 * 23, 1).to(self.device1)

    def forward(self, positions):
        flattened_float_positions = positions.flatten().to(torch.float)
        futures0 = torch.jit.fork(self.l0, flattened_float_positions.to(self.device0))
        futures1 = torch.jit.fork(self.l1, flattened_float_positions.to(self.device1))

        energy = torch.jit.wait(futures0) + torch.jit.wait(futures1).to("cuda:0")
        return energy

class DoubleModel(torch.nn.Module):
    """
    Simple model where the energy is a linear
    transformations of the positions.
    The linear model uses double precision.
    """

    def __init__(self):
        super().__init__()
        self.l0 = torch.nn.Linear(3 * 23, 1).to(torch.double).cuda()

    def forward(self, positions):
        energy = self.l0(positions.flatten())
        return energy

def run_model_no_jit(model, pdb):
    """
    Run the model before jitting it
    """
    positions = positions_from_pdb(pdb)
    model(positions)
    print("Succesfully ran model without jit")

def run_model_jit(model, pdb):
    """
    Run the model after jitting it
    """
    positions = positions_from_pdb(pdb)
    jit_model = torch.jit.script(model)
    jit_model(positions)
    print("Succesfully ran model with jit")

def run_model_loaded_jit(model, pdb):
    """
    Run the model after jitting it, saving and reloading
    """
    positions = positions_from_pdb(pdb)
    jit_model = torch.jit.script(model)
    jit_model.save("model.pt")
    del jit_model
    loaded_jit_model = torch.jit.load("model.pt")
    loaded_jit_model(positions)
    print("Succesfully ran loaded model with jit")

def run_openmm_torch(model, pdb):
    """
    Run the openmm-torch model
    """
    jit_model = torch.jit.script(model)
    jit_model.save("model.pt")
    torchforce = openmmtorch.TorchForce("model.pt")

    forcefield = ForceField("amber14-all.xml")
    system = forcefield.createSystem(pdb.topology, nonbondedMethod=NoCutoff)
    system.addForce(torchforce)
    integrator = LangevinIntegrator(300 * kelvin, 1 / picosecond, 0.5 * femtoseconds)

    simulation = Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)
    simulation.step(1)

def positions_from_pdb(pdb):
    """
    Return a tensor with the positions
    """
    positions = np.asarray(pdb.positions.value_in_unit(nanometers))
    positions = torch.tensor(positions).to(torch.double).cuda()
    positions.requires_grad_(True)
    positions.retain_grad()
    return positions

def run_models(model, pdb):
    """
    Run the four tests
    """
    run_model_no_jit(model, pdb)
    run_model_jit(model, pdb)
    run_model_loaded_jit(model, pdb)
    try:
        run_openmm_torch(model, pdb)
        print("Succesfully ran loaded model with openmm-force")
    except OpenMMException:
        print("Failed to run loaded model with openmm-force")


if __name__ == "__main__":
    pdb = PDBFile("diala.pdb")
    double_model = DoubleModel()
    print("Starting double test")
    run_models(double_model, pdb)
    single_gpu_model = ParallelModel("cuda:0", "cuda:0")
    print("Starting single gpu test")
    run_models(single_gpu_model, pdb)
    multi_gpu_model = ParallelModel("cuda:0", "cuda:1")
    print("Starting multi gpu test")
    run_models(multi_gpu_model, pdb)
