import os


def parse_configuration(config_file):

    with open(config_file, 'r') as f:
        lines = f.readlines()

    lines = [
        line.strip()
        for line in lines
        if line.strip() and not line.startswith('#')
        ]

    configuration = []
    for line in lines:
        if line.startswith('[') and line.endswith(']'):
            module = {}
            configuration.append(module)
            module['type'] = line[1:-1].strip()
        else:
            key, value = (i.strip() for i in line.split("="))
            module[key] = value

    return configuration
