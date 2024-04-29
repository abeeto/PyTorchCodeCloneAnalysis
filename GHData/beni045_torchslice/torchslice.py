import torch
import torch.fx
from torch.fx.node import Node

from typing import Dict, Tuple, Union


@torch.fx.wrap
def set_input(x):
    """
    Set new input for slicing the model.
    Usage:
    foo = set_input(foo)
    """
    return x


@torch.fx.wrap
def set_output(x):
  """
  Set new output for slicing the model.
  Usage:
    foo = set_output(foo)
  --- All other outputs will be removed!!---
  """
  return x



class GraphRunner:
    """
    --- Modified ShapeProp class from torch.fx examples: https://pytorch.org/docs/stable/fx.html#the-interpreter-pattern ---
    This module takes in a torch.nn.Module at initializtion.
    This module can run each node in the graph, so intermediate outputs can be saved.
    The inputs and outputs (dicts) defined with set_input(), set_output() functions are returned
    when running the self.propagate method.
    """
    def __init__(self, mod):
        self.mod = mod
        self.graph = mod.graph
        self.modules = dict(self.mod.named_modules())

    def propagate(self, *args) -> Tuple[Dict, Dict]:
        args_iter = iter(args)
        env : Dict[str, Node] = {}

        def load_arg(a):
            return torch.fx.graph.map_arg(a, lambda n: env[n.name])

        def fetch_attr(target : str):
            target_atoms = target.split('.')
            attr_itr = self.mod
            for i, atom in enumerate(target_atoms):
                if not hasattr(attr_itr, atom):
                    raise RuntimeError(f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}")
                attr_itr = getattr(attr_itr, atom)
            return attr_itr

        # Store new IO
        self.new_inputs = {}
        self.new_outputs = {}

        for node in self.graph.nodes:
            # Run node
            if node.op == 'placeholder':
                result = next(args_iter)
                self.new_inputs[node.name] = result  # save original input
            elif node.op == 'get_attr':
                result = fetch_attr(node.target)
            elif node.op == 'call_function':
                result = node.target(*load_arg(node.args), **load_arg(node.kwargs))
            elif node.op == 'call_method':
                self_obj, *args = load_arg(node.args)
                kwargs = load_arg(node.kwargs)
                result = getattr(self_obj, node.target)(*args, **kwargs)
            elif node.op == 'call_module':
                result = self.modules[node.target](*load_arg(node.args), **load_arg(node.kwargs))
            env[node.name] = result

            # Save IO
            if 'set_input' in node.name:
                self.new_inputs[node.args[0].name] = result
            elif 'set_output' in node.name:
                self.new_outputs[node.args[0].name] = result
      
        return (self.new_inputs, self.new_outputs)
     


def transform(m : torch.nn.Module, all_inputs=None) -> Tuple[torch.fx.GraphModule, Union[Dict,None]]:
    # Convert nn.Module to fx.GraphModule
    gm : torch.fx.GraphModule = torch.fx.symbolic_trace(m)  

    # Store new output node args, orig output node
    new_outputs = []
    orig_output = None

    # Go through all the nodes in the Graph
    for n in gm.graph.nodes:
        # Need to swapout dummy input node (function) with placeholder
        if 'set_input' in n.name:
            # Set the insert point, add the new node, and replace all uses
            # of `n` with the new node
            with gm.graph.inserting_after(n):
                new_node = gm.graph.create_node('placeholder', n.args[0].name)  #(torch.bitwise_and, n.args, n.kwargs)
                n.replace_all_uses_with(new_node)

            # Remove the old node from the graph
            gm.graph.erase_node(n)

        # Store output nodes to use as args for real output node
        elif 'set_output' in n.name:
            new_outputs.append(n.args[0])

        # Store original output node to remove later
        elif n.op == 'output':
            orig_output = n
            
    if len(new_outputs):
        gm.graph.output(tuple(new_outputs))
        gm.graph.erase_node(orig_output)


    gm.graph.eliminate_dead_code()

    # Unused placeholders (inputs) are not removed by elim_dead_code()
    # so remove them manually
    for node in gm.graph.nodes:
        if node.op == 'placeholder' and node.users == {}:
            if all_inputs is not None and node.name in all_inputs:
                del all_inputs[node.name]
            gm.graph.erase_node(node)

    gm.recompile()
    gm.graph.lint()
  
    return gm, all_inputs


def slice_module(mod : torch.nn.Module, mod_inputs : Tuple[torch.tensor,...] = None):
    # No inputs provided, cant return intermediate IO
    if mod_inputs is None:
        sliced_mod, _ = transform(mod)

        return sliced_mod, None, None

    # Inputs provided, can return intermdiate IO
    else:
        gm_mod = torch.fx.symbolic_trace(mod)
        gr = GraphRunner(gm_mod)
        inputs, outputs = gr.propagate(*mod_inputs)
        sliced_mod, inputs = transform(mod, inputs)

        return sliced_mod, inputs, outputs