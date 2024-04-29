import matplotlib.pyplot as plt
import numpy as np
import importlib

from pydrake.solvers.mathematicalprogram import MathematicalProgram, Solve
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.analysis import Simulator, SimulatorStatus
from pydrake.solvers.mathematicalprogram import SolutionResult

# from pydrake.all import (
#     DiagramBuilder,
#     Simulator,
#     SolutionResult
# )

from pydrake.multibody.tree import (
    JointActuatorIndex
)

from pydrake.geometry import SceneGraph
from pydrake.common import FindResourceOrThrow
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.parsing import Parser
from pydrake.trajectories import PiecewisePolynomial
from pydrake.solvers.snopt import SnoptSolver

import time
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.analysis import Simulator
from pydrake.systems.drawing import plot_system_graphviz

# from pydrake.all import (
#     ConnectMeshcatVisualizer,
#     # DiagramBuilder,
# #     Simulator,
# #     SimulatorStatus
# )

from pydrake.geometry import (
    SceneGraph, ConnectDrakeVisualizer
)
from pydrake.systems.meshcat_visualizer import ConnectMeshcatVisualizer

from pydrake.common import FindResourceOrThrow
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.parsing import Parser
from pydrake.systems.rendering import MultibodyPositionToGeometryPose
from pydrake.systems.primitives import (
    TrajectorySource,
    Demultiplexer,
    ConstantVectorSource
)

import argparse

import meshcat
import meshcat.geometry as geom
import meshcat.transformations as tforms

# from obstacles import Obstacles
# from viz_helper import *
# from helper import *
# from constraints import *
