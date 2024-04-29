import torch
import os
import ujson
import shutil


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def ensure_folder(filename):
    folder = os.path.dirname(os.path.abspath(filename))
    mkdir(folder)


def rmdir(directory):
    try:
        shutil.rmtree(directory)
    except:
        pass
        

def save_json(filename, obj):
    ensure_folder(filename)
    with open(filename, "w", encoding='utf8') as file:
         ujson.dump(obj, file, indent=4, ensure_ascii=False)


def load_json(filename):
    with open(filename, 'r', encoding='utf8') as file:
        return ujson.load(file)


def read_all_lines(filename, encoding='utf8'):
    if not os.path.isfile(filename):
        return []
    with open(filename, encoding=encoding) as file:
        for line in file:
            line = line.rstrip().strip('\ufeff')
            if line:
                yield line


def write_all_lines(filename, lines, encoding='utf8'):
    ensure_folder(filename)
    with open(filename, 'w', encoding=encoding) as file:
        for line in lines:
            file.write(line + '\n')


def distinct_all_lines(filename, encoding='utf8'):
    lines = set(read_all_lines(filename, encoding))
    write_all_lines(filename, sorted(lines))
    return lines


