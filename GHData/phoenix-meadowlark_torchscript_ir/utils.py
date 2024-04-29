import os
import re
import textwrap
from typing import Hashable, List, Tuple, Sequence

import pygments
import pygments.lexers
import pygments.formatters
import torch

# Might be useful for windows, hard for me to say.
normalize_sep = lambda s: os.sep.join(s.split("/"))
PATH_FILTERS = [
    os.environ["HOME"] + os.sep if "HOME" in os.environ else "",
    "/usr/",
    normalize_sep(".venv/torchscript_ir/"),
    normalize_sep(r"lib/python3.[0-9]+/site-packages/"),
    normalize_sep(".cache/"),
]


def highlight(code):
  # Useful for debugging.
  return pygments.highlight(str(code), pygments.lexers.PythonLexer(),
                            pygments.formatters.TerminalFormatter(bg='dark'))


def clean_references(graph: str) -> str:
  for pattern in PATH_FILTERS:
    graph = re.sub(pattern, "", graph)
  return graph


def demangle(graph: str) -> str:
  # Used to filter unique submodule graphs.
  return re.sub(".___torch_mangle_[0-9]+", "", graph)


def _get_module_name(module: torch.nn.Module) -> str:
  if hasattr(module, 'original_name'):
    return module.original_name
  else:
    return type(module).__name__


def _get_cleaned_method_graphs(module: torch.nn.Module) -> List[Tuple[str]]:
  methods_and_graphs = []

  for attr in sorted(set(dir(module)) - set(["__class__"])):
    try:
      value = getattr(module, attr)
      if hasattr(value, 'graph'):
        graph = demangle(clean_references(str(value.graph)))
        methods_and_graphs.append((attr, graph))
    except:
      continue

  if not len(methods_and_graphs):
    assert not hasattr(module, "graph_for_types")
    return [("*", "module had no methods with graph attrs.\n")]
  else:
    return methods_and_graphs


def _get_module_graphs(module: str) -> List[Tuple[str]]:
  name = _get_module_name(module)
  methods_and_graphs = _get_cleaned_method_graphs(module)
  graphs = [(name, method, graph) for method, graph in methods_and_graphs]
  return graphs


def get_submodule_graphs(module: str) -> List[Tuple[str]]:
  graphs = []
  graphs.extend(_get_module_graphs(module))
  for submodule in module.children():
    graphs.extend(get_submodule_graphs(submodule))
  return graphs


def filter_unique(values: Sequence[Hashable]) -> List[Hashable]:
  seen = set()
  unique_values = []
  for value in values:
    if value not in seen:
      unique_values.append(value)
      seen.add(value)
  return unique_values


def group_by_graph(graphs):
  graphs_to_names_and_methods = {}
  for name, method, graph in graphs:
    if graph not in graphs_to_names_and_methods:
      graphs_to_names_and_methods[graph] = []
    graphs_to_names_and_methods[graph].append((name, method))

  grouped_graphs = []
  for graph, names_and_methods in graphs_to_names_and_methods.items():
    names_and_methods = [
        f"{name}.{method}" for name, method in names_and_methods
    ]
    names_and_methods = filter_unique(names_and_methods)
    graph = textwrap.indent(graph, '  ')
    grouped_graphs.append("\n".join(names_and_methods + [graph]))
  return grouped_graphs


def save_tsir(model, jit_model, model_path, save_model=False):
  os.makedirs(model_path, exist_ok=True)
  if issubclass(jit_model.__class__, torch.jit.TracedModule):
    jit_type = "trace"
  else:
    jit_type = "script"

  with open(os.path.join(model_path, f"inlined.{jit_type}.tsir"), "w") as f:
    f.write(clean_references(str(jit_model.inlined_graph)))

  graphs = get_submodule_graphs(jit_model)
  graphs = group_by_graph(graphs)
  lines = str(model).split("\n")
  lines.append("")
  lines.extend(graphs)

  with open(os.path.join(model_path, f"submodules.{jit_type}.tsir"), "w") as f:
    f.write("\n".join(lines) + "\n")

  if save_model:
    torch.jit.save(jit_model,
                   os.path.join(model_path, f"{model_name}.{jit_type}.pt"))
