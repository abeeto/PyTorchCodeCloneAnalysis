import graphviz
import torch
import torch.autograd as autograd
import torch.nn as nn


def number_of_parameters(module: nn.Module):
    """
    Calculate the total number of parameters in a module
    """
    return sum(parameter.numel() for parameter in module.parameters())


def inspect_module(module, v):
    """
    Draw the graph of a neural network. Since Pytorch builds the networks dynamically,
    there's no statical structure of a network. The only way to draw a network is to
    feed a variable into it and track its trace through the network. This function
    puts hooks in the module to track the sequence they are called. 

    Parameters
    ==========
    module: torch.nn.Module
        The neural network to be drawn
    v: torch.autograd.Variable
        The variable to be fed to the module
    
    Returns
    =======
    graphviz.Digraph
    """
    hooks = []
    nodes = {}
    trash = [] # Keep the intermediate variables alive, so that their ids won't be reused

    def create_node(id, name, dim, **kwargs):
        nodes[id] = {
            "name": name,
            "kwargs": kwargs,
            "edges": set(),
            "dim": dim,
        }

    def handle_variable(variable, parent_id):
        trash.append(variable)
        v_id = var_to_id(variable)
        if v_id not in nodes:
            grad_fn = variable.grad_fn
            if grad_fn is None:
                create_node(v_id, "Variable"+str(tuple(variable.size())), variable.size(-1), fillcolor="orange")
            else:
                create_node(v_id, str(type(grad_fn).__name__).replace("Backward", ""), variable.size(-1), fillcolor="green")
        nodes[v_id]["edges"].add(parent_id)

    def generate_graph():
        node_attr = dict(style='filled',
                 shape='box',
                 align='left',
                 fontsize='12',
                 ranksep='0.1',
                 height='0.2')
        g = graphviz.Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"), format="png")
        while 1:
            # Remove the isolated sub-graphs
            for key in nodes.keys():
                if not nodes[key]["edges"] and nodes[key]["dim"] != 1:
                    del nodes[key]
                    for key2 in nodes.keys():
                        if key in nodes[key2]["edges"]:
                            nodes[key2]["edges"].remove(key)
                    break
            else:
                break
        for key, value in nodes.items():
            g.node(key, value["name"], **value["kwargs"])
            for edge in value["edges"]:
                g.edge(key, edge)
        return g

    def var_to_id(variable):
        return str(id(variable))

    def hook(module, inputs, output):
        if not isinstance(output, autograd.Variable):
            return
        if not isinstance(inputs, tuple):
            inputs = (inputs, )
        o_id = var_to_id(output)
        module_name = str(module)
        if o_id not in nodes:
            trash.append(output)
            create_node(o_id, module_name, output.size(-1), fillcolor="lightblue")
            for input in inputs:
                if isinstance(input, torch.autograd.Variable):
                    handle_variable(input, o_id)
    for sub_module in module.modules():
        hooks.append(sub_module.register_forward_hook(hook))
    module(v)
    for hook in hooks:
        hook.remove()
    graph = generate_graph()
    return graph