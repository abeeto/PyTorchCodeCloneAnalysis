import torch
from .hooks import register_forward_hooks

from collections import defaultdict

import sys
sys.setrecursionlimit(6000) # default 3000

# def chunk_string(s, length):
#     return [s[i:length + i] for i in range(0, len(s), length)]

# def _beautify(name):
#     return '\n'.join(chunk_string(name, 20))

def _to_leaf(*args):
    return tuple([arg.detach() for arg in args])

def _collect_tensors(out):
    if torch.is_tensor(out):
        yield out
    elif isinstance(out, (tuple, list)):
        for _o in out:
            for __o in _collect_tensors(_o):
                yield __o
    elif isinstance(out, dict):
        for _o in out.values():
            for __o in _collect_tensors(_o):
                yield __o
    else:
        import warnings
        warnings.warn('{} can not be colleted as a tensor'.format(out))

def _create_tensors(shape, device='cpu'):
    if isinstance(shape, (tuple, list)):
        if isinstance(shape[0], int):
            return torch.randn(*shape, device=device)
        else:
            return list([_create_tensors(_sh) for _sh in shape])
    elif isinstance(shape, dict):
        return {k:_create_tensors(_sh) for k, _sh in shape.items()}


def _get_module_info(mod):
    # format: kxk/s/p/d:<dilation>/g:<group>
    strs = []
    try:
        if hasattr(mod, 'kernel_size'):
            strs.append('x'.join(map(str, mod.kernel_size)))
        if hasattr(mod, 'stride'):
            strs.append(str(mod.stride[0]) if mod.stride[0] == mod.stride[1] else 'x'.join(map(str, mod.stride)))
        if hasattr(mod, 'padding'):
            strs.append(str(mod.padding[0]) if mod.padding[0] == mod.padding[1] else 'x'.join(map(str, mod.padding)))
        if hasattr(mod, 'dilation'):
            if max(mod.dilation) > 1:
                strs.append('d:'+str(mod.dilation[0]) if mod.dilation[0] == mod.dilation[1] else 'x'.join(map(str, mod.dilation)))
        if hasattr(mod, 'groups'):
            if mod.groups > 1:
                strs.append('g:'+str(mod.groups))
        if hasattr(mod, 'inplace') and mod.inplace:
            strs.append('inplace')
    except Exception as e:
        print(e)
    return '/'.join(strs)


def _get_info_grad_fn(model, inputs, output):
    # trace back to construct the computational graph

    def add_node(grad_fn):
        # print(grad_fn)
        if hasattr(grad_fn, 'variable'):
            # DELEGATE AccumulateGrad to Parameter
            param = grad_fn.variable
            info['nodes'][str(id(param))].update(dict(_type='PARAMETER',
                _class=type(param).__name__, shape=list(param.shape)))
        else:
            info['nodes'][str(id(grad_fn))].update(dict(_type='BACKWARD', _class=type(grad_fn).__name__))

    seen = set()
    def add_edges(grad_fn):
        if grad_fn is None or grad_fn in seen:
            return
        add_node(grad_fn)
        seen.add(grad_fn)
        assert hasattr(grad_fn, 'next_functions')

        for gf, _ in grad_fn.next_functions:
            if gf is not None:
                if hasattr(gf, 'variable'):
                    # DELEGATE AccumulateGrad to Parameter
                    info['edges'][str((id(gf.variable)))].append(str(id(grad_fn)))
                else:
                    info['edges'][str((id(gf)))].append(str(id(grad_fn)))
                add_edges(gf)

    info = {'nodes': defaultdict(dict), 'edges': defaultdict(list), 'outputs': []}
    # info = {'nodes': dict of dict, 'edges': dict of list (graph adjacency list)}
    # info['nodes'][i] = {...}          # node_attrs
    # info['edges'][i] = [j, k, ...]    # adjacent list

    for out in _collect_tensors(output):
        add_edges(out.grad_fn)
        info['edges'][str(id(out.grad_fn))].append(str(id(out)))
        info['nodes'][str(id(out))].update(dict(_type='OUTPUT'), _class='Output', shape=list(out.shape))
        info['outputs'].append(str(id(out)))
    # NOTE: inputs and model are only used here
    for in_ in _collect_tensors(inputs):
        info['nodes'][str(id(in_))].update(dict(_type='INPUT'), _class='Input', shape=list(in_.shape))
    if model is not None:
        for name, param in model.named_parameters():
            # NOTE: can introduced not traced back parameters for MaskRCNN (defined but not used??)
            info['nodes'][str(id(param))].update(dict(_type='PARAMETER', _class=type(param).__name__, name=name, shape=list(param.shape)))
    return info

def _transfer_forward_hook_info(info, hook):
    # transfer module name, hyperparameters, output shape to info
    for _info in hook:
        mod, _in, _out = _info['module'], _info['inputs'], _info['output']
        if torch.is_tensor(_out): # NOTE: only for convential modules, i.e. output is single tensor
            # DELEGATE module (name, class, hyper-parameters), _out (output shape) to _out.grad_fn
            nid = str(id(_out.grad_fn))
            if nid in info['nodes']: # NOTE: only transfer those can be matched
                info['nodes'][nid].update(name=_info['module_name'], module_class=type(mod).__name__,
                    module_info=_get_module_info(mod), shape=list(_out.shape))

def _invert_edges(edges):
    inv_edges = defaultdict(list) # adjacient list: invert edges is to trace node parents
    for src, dsts in edges.items():
        for dst in dsts:
            inv_edges[dst].append(src)
    return inv_edges

def argsort(l):
    ll = [(e, i) for i, e in enumerate(l)]
    ll = sorted(ll)
    return [ee[1] for ee in ll]

def _transfer_node_names(info):
    nodes, edges = info['nodes'], info['edges']

    from os.path import commonprefix 

    # invert edges to trace back parents
    inv_edges = _invert_edges(edges)

    # topological sort (dfs from outputs)
    def _dfs(info, edges, root, res, visited):
        if root not in visited:
            children = edges[root] if root in edges else []
            if not len(children) == 0:
                for c in children:
                    _dfs(info, edges, c, res, visited)
            res.append(root)
            visited.add(root)

    res = []; visited = set()
    for out in info['outputs']:
        _dfs(info, inv_edges, out, res, visited)
    topo = res # topolical sorted node ids

    # if a node has no name, set its name to the common prefix of its parents
    for nid in res:
        nd = nodes[nid]
        if 'name' not in nd:
            pa = inv_edges[nid]
            pa_names = [nodes[_pa]['name'] for _pa in pa if 'name' in nodes[_pa]]
            if len(pa_names) == 1:
                nd['name'] = pa_names[0]
            elif len(pa_names) > 1:
                levels = [n.count('.') for n in pa_names]
                ind = levels.index(max(levels)) # argmax
                name = pa_names[ind].rsplit('.', 1)[0] 
                # name = commonprefix(pa_names).rstrip('.')
                nd['name'] = name + '.o' if len(pa_names) >= 2 else name

    return info, topo

def _present_graph(info, subgraph_level=-1, with_node_id=False, ignore=[], transfer_names=False):
    # info = {'nodes': dict of dict, 'edges': dict of list (graph adjacency list)}
    # info['nodes'][id] = {...}          # node_attrs
    # info['edges'][id] = [j, k, ...]    # adjacent list
    # subgraph_level: -1 (do not use subgraph), 0 (max level), >=1 (specified level)

    def _get_label(id_, nd, with_node_id):
        strs = [] if not with_node_id else [id_]
        if 'name' in nd and nd['name']:
            strs.append(nd['name'])
        if '_class' in nd:
            if nd['_class'] not in ['Parameter']: # remove Parameter as it is obvious to see from the color
                strs.append(nd['_class'].split('Backward')[0]) # remove Backward surfix
        if 'module_class' in nd:
            _extra = None
            if nd['module_class'] == 'Conv2d': # Indicate Conv2d depthwise, group, or pointwise
                if 'g:' in nd['module_info']:
                    groups = int(nd['module_info'].split('g:')[1].split('/')[0])
                    _extra = 'Depthwise' if groups == nd['shape'][1] else 'Group'
                elif '1x1' in nd['module_info']:
                    _extra = 'Pointwise'

            if _extra is not None:
                strs.append('[{} ({})]'.format(nd['module_class'], _extra))
            else:
                strs.append('[{}]'.format(nd['module_class']))
   
        # if '_type' in nd:
        #     strs.append('({})'.format(nd['_type'])) # already indicated in color
        if 'module_info' in nd and nd['module_info']:
            strs.append(nd['module_info'])
        if 'shape' in nd:
            strs.append('(' + ','.join(map(str, nd['shape'])) + ')')
        return '\n'.join(strs)

    # def _graph_split(info, sub):
    #     sub = set(sub)
    #     info1 = dict(nodes=defaultdict(list), edges=defaultdict(list))
    #     info2 = dict(nodes=defaultdict(list), edges=defaultdict(list))

    #     for id_, nd in info['nodes'].items():
    #         if id_ in sub:
    #             info1['nodes'][id_] = nd

    #     for src, dsts in info['edges'].items():
    #         for dst in dsts:
    #             if src in sub and dst in sub:
    #                 info1['edges'][src].append(dst)
    #             else:
    #                 info2['edges'][src].append(dst)
    #                 info2['nodes'][src] = info['nodes'][src]
    #                 info2['nodes'][dst] = info['nodes'][dst]
    #     return info1, info2

    def _dfs_all_paths(G,v,seen=None,path=None):
        if seen is None: seen = set()
        if path is None: path = [v]

        seen.add(v)

        paths = []
        for t in G[v]:
            if t not in seen:
                t_path = path + [t]
                paths.append(tuple(t_path))
                paths.extend(_dfs_all_paths(G, t, seen, t_path))
        return paths

    def _gen_dot(g, info):
        all_paths = []
        inv_edges = _invert_edges(info['edges'])
        for output in info['outputs']:
            all_paths.extend(_dfs_all_paths(inv_edges, output))
        all_paths = all_paths[::-1] # optional (straighten Input)

        ind = argsort([len(p) for p in all_paths])[-1]
        longest_path = all_paths[ind]
        longest_path_edges = set([(src, dst) for src, dst in zip(longest_path[:-1], longest_path[1:])])

        for id_, nd in info['nodes'].items():
            g.node(id_, label=_get_label(id_, nd, with_node_id), **STYLES[nd['_type'] if '_type' in nd else 'UNKNOWN'])
        for src, dsts in info['edges'].items():
            for dst in dsts:
                if (src, dst) in longest_path_edges or (dst, src) in longest_path_edges:
                    g.edge(src, dst, weight='5')
                else:
                    g.edge(src, dst)
                # if '_class' in info['nodes'][src] and info['nodes'][src]['_class'] in ['Parameter', 'TBackward']:
                #     g.edge(src, dst)
                # else:
                #     g.edge(src, dst, weight='5') # make non-parameter edges large weight to straighten the graph
        return g

    def _get_children(subgs, prefix):
        children = [k for k, v in subgs.items() if k.startswith(prefix) and k != prefix and k[len(prefix)] == '.']
        level = prefix.count('.') + 1
        children = list(set(['.'.join(c.split('.')[:level+1]) for c in children]))
        return children

    # NOTE: subgraph with only one node is not necessary, merge to parent
    def _is_leaf(name, subgs):
        children = [n for n in subgs.keys() if n.startswith(name)]
        return len(children) == 1

    def _gen_subgraph(g, subgs, prefix, i):
        nids = subgs[prefix]
        children = _get_children(subgs, prefix)
        # print(prefix, children)
        with g.subgraph(name='cluster_{}'.format(i[0])) as c: # NOTE: subgraph name must be cluster_<int>
            i[0] += 1
            c.attr(style='dotted')
            for nid in nids:
                c.node(nid)
            c.attr(label=prefix)
            for name in children:
                _gen_subgraph(c, subgs, name, i)

    if transfer_names:
        # infer node names if missing
        info, _ = _transfer_node_names(info)

    # draw dot graph
    import graphviz
    g = graphviz.Digraph(node_attr=STYLES['DEFAULT'])
    # g.attr(splines='false')
    g = _gen_dot(g, info)

    # draw subgraph
    ignore = set(ignore)
    if subgraph_level >= 0:
        # 1) construct
        subgs = defaultdict(list)
        for id_, nd in info['nodes'].items():
            if 'name' in nd:
                sub = '.'.join(nd['name'].split('.')[:subgraph_level]) if subgraph_level >= 1 else nd['name']
                subgs[sub].append(id_)
            else:
                subgs[''].append(id_)
        # 2) remove subgraph has only one node, in ignore or .weight/.bias
        for name, nids in list(subgs.items()):
            if name in ignore or name.endswith('.weight') or name.endswith('.o') or (len(nids) <= 1 and _is_leaf(name, subgs)):
                ns = name.rsplit('.', 1)
                new_name = ns[0] if len(ns) == 2 else ''
                subgs[new_name].extend(nids)
                del subgs[name]
        # from pprint import pprint
        # pprint(subgs)
        # 3) draw
        i = [0]
        # prefixs = [k for k in subgs.keys() if k and '.' not in k and k not in ignore]
        prefixs = list(set([k.split('.', 1)[0] for k in subgs.keys() if k]))
        for _ig in ignore:
            prefixs.extend(list(_get_children(subgs, _ig)))
        prefixs = set(prefixs).difference(ignore)
        # print(prefixs)
        for prefix in prefixs:
            _gen_subgraph(g, subgs, prefix, i)

    return g

def _backup_requires_grad(model, inputs):
    # record requires_grad status for inputs and parameters
    # NOTE: requires_grad can only be changed for leaf and float tensors
    params_status = {}
    for t in _collect_tensors(inputs):
        params_status[t] = t.requires_grad
        t.requires_grad_(True)
    for n, t in model.named_parameters():
        params_status[t] = t.requires_grad
        t.requires_grad_(True)
    return params_status

def _recover_requires_grad(params_status):
    # recover requires_grad
    for t, req_grad in params_status.items():
        try:
            t.requires_grad_(req_grad)
        except Exception as e:
            t.data.requires_grad_(req_grad)


STYLES = dict(DEFAULT=dict(shape='box'), # default style is applied to all nodes, default fontsize: 14
    INFO=dict(style='filled', fillcolor='slateblue'),
    INPUT=dict(style='filled', fillcolor='orchid'),
    OUTPUT=dict(style='filled', fillcolor='darkorchid'),
    INTERMEDIATE=dict(),
    PARAMETER=dict(style='filled', fillcolor='orange', fontsize='9'),
    PARAMETER_FREEZED=dict(style='filled', fillcolor='bisque', fontsize='9'), # light orange
    MODULE=dict(style='filled'), # gray
    GRADFN=dict(style='filled'),
    BACKWARD=dict(style='filled'),
    UNKNOWN=dict(style='filled', fillcolor='red'),)
    # Backward is the same as GradFn, Backward is traced back from out.grad_fn.next_functions,
    # GradFn is from tensor.grad_fn (in forward hooks)


def plot_network(model, *inputs, output=None, subgraph_level=-1, with_node_id=False, ignore=[], transfer_names=True, **kwargs):
    # NOTE: 1. str(id(grad_fn)) can be changed when trace back output.grad_fn
    #          must trace back out.grad_fn first, and then call str(id(tensor.grad_fn)) can keep. WHY??
    # 2. inputs (GradFn) -> Module -> output(s) (GradFn) (forward hook)
    #         |                          ^
    #         |--------------------------|
    #             out.grad_fn
    #   DELEGATE Module, Output Tensor to Output Tensor GradFn
    # 3. AccumulateGrad (BackwardFn) -> Parameter
    #   DELEGATE AccumulateGrad to Parameter
    all_inputs = inputs + tuple(_collect_tensors(kwargs))
    total_flops = 0
    if output is None:
        params_status = _backup_requires_grad(model, all_inputs) # also set_requires_grad to True to trace back
        with register_forward_hooks(model) as forward:
            model.eval()
            with torch.set_grad_enabled(True):
                output = model(*inputs, **kwargs)
            from ._summary import _flops_full
            forward.register_extra_hook('flops_full', _flops_full)
            total_flops = sum([_info['flops_full'] for _info in forward])
            info = _get_info_grad_fn(model, all_inputs, output)
            _transfer_forward_hook_info(info, forward) # optional (but for more module info)
        _recover_requires_grad(params_status)
    else:
        info = _get_info_grad_fn(model, all_inputs, output)

    # update for freezed parameters
    for name, param in model.named_parameters():
        if not param.requires_grad:
            info['nodes'][str(id(param))].update(dict(_type='PARAMETER_FREEZED',
                _class=type(param).__name__, name=name))
    g = _present_graph(info, subgraph_level, with_node_id, ignore=ignore, transfer_names=transfer_names)
    _info = ['Flops: {:,}'.format(total_flops), 'Params: {:,}'.format(sum(p.numel() for p in model.parameters()))]
    g.node('flops_and_params', label='\n'.join(_info), **STYLES['INFO'])
    return g


# for plot_network_callback, if both networks (possible multiple) and inputs are constructed inside callback
# need to manually add them inside the code, especially many levels of callbacks (e.g. NeRF).
# they are cleared at the end of plot_network_callback
# MODELS = []
INPUTS = []

# aux functions
def add_inputs(*args):
    for arg in args:
        if torch.is_tensor(arg):
            INPUTS.append(arg)
            arg.requires_grad = True

def clear_inputs():
    INPUTS.clear()


def plot_network_callback(callback, kwargs, models=[], inputs=[], subgraph_level=-1, with_node_id=False, ignore=[], transfer_names=True):
    # models = models + MODELS (MODELS must be pre-collected outside out callback
    if isinstance(models, torch.nn.Module): # single model
        model = models
    else:
        model = torch.nn.Module() # mutiple models (e.g. NeRF has network_fn, network_fine, embed_pts, embed_dirs
        for i, mod in enumerate(models):
            setattr(model, 'net{}'.format(i), mod)
        
    params_status = _backup_requires_grad(model, inputs) # also set_requires_grad to True to trace back
    with register_forward_hooks(model) as forward:
        model.eval()
        with torch.set_grad_enabled(True):
            output = callback(**kwargs)
        inputs = inputs + INPUTS # additional INPUTS collected inside callback, additional MODELS collected inside callback is too late (should before registering hooks)
        info = _get_info_grad_fn(model, inputs, output)
        _transfer_forward_hook_info(info, forward) # optional (but for more module info)
    _recover_requires_grad(params_status)

    # update for freezed parameters
    for name, param in model.named_parameters():
        if not param.requires_grad:
            info['nodes'][str(id(param))].update(dict(_type='PARAMETER_FREEZED',
                _class=type(param).__name__, name=name))
    g = _present_graph(info, subgraph_level, with_node_id, ignore=ignore, transfer_names=transfer_names)
    # Cleanup
    # MODELS.clear()
    INPUTS.clear()
    return g


def plot_network_tensorboard(model, *inputs, writer=None):
    model.eval()

    if writer is not None:
        writer.add_graph(model, inputs)
    else:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter()
        writer.add_graph(model, inputs)
        writer.close()


def test_plot_network():

    # ResNet50
    import torch
    import torchvision.models as models

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = models.resnet18().to(device)
    inputs = torch.randn(1, 3, 224, 224).to(device)

    plot_network(model, inputs).save('resnet18.gv')
    plot_network(model, inputs, subgraph_level=0).save('resnet18_grouped.gv')

    output = model(inputs)
    plot_network(model, inputs, output=output).save('resnet18_grad_fn.gv')


    # ViT
    import torch
    import timm.models as models

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = models.vit_base_patch16_224().to(device)
    inputs = torch.randn(1, 3, 224, 224).to(device)

    plot_network(model, inputs).save('vit_base_patch16_224.gv')
    ignore = ['blocks'] + ['blocks.{}.drop_path'.format(i) for i in range(12)]
    plot_network(model, inputs, subgraph_level=0, ignore=ignore).save('vit_base_patch16_224_grouped.gv')
    
def test_plot_network_callback():
    import torch
    import torchvision.models as models

    device = torch.device('cpu') if torch.cuda.is_available() else torch.device('cpu')
    model = models.resnet18().to(device)
        
    def callback(model, size):
        inputs = torch.randn(1, 3, size, size).to(device)
        output = model(inputs)
        return output
    
    kwargs = dict(model=model, size=384)
    models = [model]
    plot_network_callback(callback, kwargs, models=[model], transfer_names=False).save('resnet18_callback.gv')

def test_plot_network_callback2():
    import torch
    import torchvision.models as models

    device = torch.device('cpu') if torch.cuda.is_available() else torch.device('cpu')

    # NOTE: models must be pre-collected for registration to infer output_shape and module_name info, otherwise will not show such info
    def callback2(size=224):
        model = models.resnet18().to(device)
        inputs = torch.randn(1, 3, size, size).to(device)
        ##### model and inputs are not accessible from outside, need to insert codes to keep track of them for viz
        # MODELS.append(model)
        INPUTS.append(inputs)
        ##### end
        output = model(inputs)
        return output

    plot_network_callback(callback2, kwargs={}, transfer_names=False).save('resnet18_callback2.gv')
    

def test_plot_network_maskrcnn():

    # MaskRCNN
    # 1) random input
    import torch
    import torchvision.models as models

    # model = models.detection.maskrcnn_resnet50_fpn(pretrained=False)
    # model.eval()
    # x = [torch.rand(3, 300, 400)]
    # # predictions = model(x)
    # plot_network(model, x).save('maskrcnn_resnet50_fpn_random.gv')

    # 2) real input
    # https://pytorch.org/vision/master/auto_examples/plot_visualization_utils.html
    from .utils import imread_url
    from torchvision.transforms.functional import convert_image_dtype
    import numpy as np

    url = 'https://farm3.staticflickr.com/2373/2421365812_cda8476bb4_z.jpg' 
    img = torch.tensor(np.array(imread_url(url)).transpose(2,0,1), dtype=torch.uint8) # (3, H, W)
    batch = convert_image_dtype(torch.stack([img]), dtype=torch.float)

    model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    plot_network(model, batch).save('maskrcnn_resnet50_fpn.gv')
    plot_network(model, batch, subgraph_level=0, ignore=['backbone', 'backbone.body']).save('maskrcnn_resnet50_fpn_grouped.gv')

if __name__ == '__main__':
    
    test_plot_network()
    test_plot_network_maskrcnn()
    test_plot_network_callback()
    test_plot_network_callback2()
