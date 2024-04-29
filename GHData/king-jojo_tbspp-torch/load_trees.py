"""Parse trees from a data source."""
import ast
import sys
import pickle
import random
from collections import defaultdict

def parse(infile, outfile, label_key, maxsize, minsize, test):
    """Parse trees with the given arguments."""
    print ('Loading pickle file')

    sys.setrecursionlimit(1000000)
    with open(infile, 'rb') as file_handler:
        data_source = pickle.load(file_handler)

    print('Pickle file load finished')

    train_samples = []
    test_samples = []

    train_counts = defaultdict(int)
    test_counts = defaultdict(int)

    for item in data_source:
        root = item['tree']
        label = item['metadata'][label_key]
        sample, size = _traverse_tree(root)

        if size > maxsize or size < minsize:
            continue

        roll = random.randint(0, 100)

        datum = {'tree': sample, 'label': label}

        if roll < test:
            test_samples.append(datum)
            test_counts[label] += 1
        else:
            train_samples.append(datum)
            train_counts[label] += 1

    random.shuffle(train_samples)
    random.shuffle(test_samples)
    # create a list of unique labels in the data
    labels = list(set(train_counts.keys() + test_counts.keys()))
    print('Dumping sample')
    with open(outfile, 'wb') as file_handler:
        pickle.dump((train_samples, test_samples, labels), file_handler)
        file_handler.close()
    print('dump finished')
    print('Sampled tree counts: ')
    print('Training:', train_counts)
    print('Testing:', test_counts)

def _traverse_tree(root):
    num_nodes = 1
    queue = [root]
    root_json = {
        "node": _name(root),

        "children": []
    }
    queue_json = [root_json]
    while queue:
        current_node = queue.pop(0)
        num_nodes += 1
        # print (_name(current_node))
        current_node_json = queue_json.pop(0)


        children = list(ast.iter_child_nodes(current_node))
        queue.extend(children)
        for child in children:
            child_json = {
                "node": _name(child),
                "children": []
            }

            current_node_json['children'].append(child_json)
            queue_json.append(child_json)

    return root_json, num_nodes

def _name(node):
    return type(node).__name__

if __name__ == '__main__':
    input_file = './data/algorithms.pkl'
    output_file = './data/algorithm_trees.pkl'
    label_key = 'label'
    max_size = 2000
    min_size = 100
    test_number = 30
    parse(input_file, output_file, label_key, max_size, min_size, test_number)

