""" Network architecture visualizer using graphviz """
import sys
from graphviz import Digraph
import genotypes as gt

def subplot(genotype, prefix, dag_layers):
    edge_attr = {
        'fontsize': '15',
        'fontname': 'Sans'
    }
    node_attr = {
        'style': 'filled',
        'shape': 'rect',
        'align': 'center',
        'fontsize': '20',
        'height': '0.5',
        'width': '0.5',
        'penwidth': '2',
        'fontname': 'Sans'
    }
    g = Digraph(
        format='png',
        edge_attr=edge_attr,
        node_attr=node_attr,
        engine='dot')
    g.body.extend(['rankdir=LR'])
    # input nodes
    n_input = dag_layers.n_input
    g_in = []
    for i in range(n_input):
        node = str(prefix)+'in_'+str(i)
        g_in.append(node)
        g.node(node, fillcolor='darkseagreen2')

    # intermediate nodes
    n_nodes = len(genotype)
    g_nodes = []
    for i in range(n_nodes):
        node = str(prefix)+'n_'+str(i)
        g.node(node, fillcolor='lightblue')
        g_nodes.append(node)

    j = 0
    for i, edges in enumerate(genotype):
        for g_child, sidx, n_state in edges:
            v = g_nodes[n_state-n_input]
            if isinstance(g_child[0], str):
                op = g_child[0]
            else:
                p_child, n_in, n_out = subplot(g_child, str(prefix)+str(j)+'_', dag_layers.edges[0])
                g.subgraph(p_child)
                g.edge(n_out, v, label='', fillcolor="gray")
                j=j+1
            
            for i, si in enumerate(sidx):
                if si < n_input:
                    u = g_in[si]
                else:
                    u = g_nodes[si-n_input]
                if isinstance(g_child[0], str):
                    g.edge(u, v, label=op, fillcolor="gray")
                else:
                    g.edge(u, n_in[i], label='', fillcolor="gray")
            
    # output node
    g_out = str(prefix)+'out'
    g.node(g_out, fillcolor='palegoldenrod')
    for i in dag_layers.merge_out_range:
        if i < n_input:
            u = g_in[i]
        else:
            u = g_nodes[i-n_input]
        g.edge(u, g_out, fillcolor="gray")
    return g, g_in, g_out

def plot(genotype, dag_layers, file_path, caption=None):
    """ make DAG plot and save to file_path as .png """
    edge_attr = {
        'fontsize': '15',
        'fontname': 'Sans'
    }
    node_attr = {
        'style': 'filled',
        'shape': 'rect',
        'align': 'center',
        'fontsize': '20',
        'height': '0.5',
        'width': '0.5',
        'penwidth': '2',
        'fontname': 'Sans'
    }
    g = Digraph(
        format='png',
        edge_attr=edge_attr,
        node_attr=node_attr,
        engine='dot')
    g.body.extend(['rankdir=TB'])

    g_child, g_in, g_out = subplot(genotype,'', dag_layers)
    for n in g_in:
        g.edge('input',n,label='',fillcolor='gray')
    g.subgraph(g_child)

    # add image caption
    if caption:
        g.attr(label=caption, overlap='false', fontsize='20', fontname='Sans')

    try:
        g.render(file_path, view=False)
        print('plot saved to: {}'.format(file_path))
    except Exception as e:
        print('render failed: {}'.format(str(e)))
        with open(file_path, 'w') as f:
            f.write(g.source)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError("usage:\n python {} GENOTYPE".format(sys.argv[0]))

    genotype_str = sys.argv[1]
    try:
        genotype = gt.from_str(genotype_str)
    except AttributeError:
        raise ValueError("Cannot parse {}".format(genotype_str))

    plot(genotype.normal, "normal")
    plot(genotype.reduce, "reduction")
