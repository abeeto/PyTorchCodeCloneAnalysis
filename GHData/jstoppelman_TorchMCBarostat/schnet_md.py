#!/usr/bin/env python
import schnetpack as sch
import os
from schnetpack.md import System, MaxwellBoltzmannInit
from schnetpack.md.calculators import SchnetPackCalculator
from schnetpack.md.simulation_hooks import logging_hooks, thermostats
from schnetpack.md.neighbor_lists import TorchNeighborList
import torch
from mc_barostat import MonteCarloAnisotropicBarostat, MonteCarloBarostat

md_system = System(1, device='cuda')
md_system.load_molecules_from_xyz("input_iso.xyz")

md_init = MaxwellBoltzmannInit(290, remove_translation=True, remove_rotation=True)
md_init.initialize_system(md_system)

#barostat = barostats.NHCBarostatIsotropic(1000, 300, 100, time_constant_barostat=300)
md_integrator = sch.md.integrators.VelocityVerlet(1.0)
#md_integrator = sch.md.integrators.VelocityVerlet(1.0)

md_model = torch.load("best_model_1103", map_location='cuda').to('cuda')

md_calc = SchnetPackCalculator(md_model, required_properties=[sch.Properties.energy, sch.Properties.forces, sch.Properties.stress], force_handle=sch.Properties.forces, position_conversion='A', force_conversion='eV/A', neighbor_list=TorchNeighborList, cutoff=4.0, cutoff_shell=0.15, stress_handle='stress')

barostat = MonteCarloAnisotropicBarostat(1000, 290, 300)

simulation_hooks = [barostat]
lang = thermostats.LangevinThermostat(290, 100)
simulation_hooks.append(lang)
#simulation_hooks = []

log_file = "nptschnet_2/sim_300.hdf5"
if os.path.exists("nptschnet_2/sim_300.hdf5"): os.remove("nptschnet_2/sim_300.hdf5")
if os.path.exists("nptschnet_2/sim.chk"): os.remove("nptschnet_2/sim.chk")

buffer_size = 100
data_streams = [logging_hooks.MoleculeStream(), logging_hooks.PropertyStream()]
file_logger = logging_hooks.FileLogger(log_file, buffer_size, data_streams = data_streams)
simulation_hooks.append(file_logger)

chk_file = "nptschnet_2/sim.chk"
checkpoint = logging_hooks.Checkpoint(chk_file, every_n_steps=100)
simulation_hooks.append(checkpoint)
md_sim = sch.md.Simulator(md_system, md_integrator, md_calc, simulator_hooks=simulation_hooks)
md_sim.simulate(1000000)
