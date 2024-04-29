import argparse
import logging
import os
import sys

parser = argparse.ArgumentParser(description="Analyse torch ops from map file")
parser.add_argument("src_file",  default=None, type=str, help='path of the source file')
parser.add_argument('-o', dest="out_file", default='out.txt', type=str, help='path of the output file')

#logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s')
#logger = logging.getLogger()
#logger_all = logging.getLogger('all')

def read_src_file(src_file):
    # file not exist, exit
    if not os.path.exists(src_file):
        print("%s does not exist!" % src_file)
        sys.exit(0)

    with open(src_file, 'r') as f:
        mark = "-----"
        line = f.readline()
        ret = {}
        if line.find(mark) == -1:
            print("the format of mapping file is wrong!")

        while True:
            line = f.readline()
            if line.find(mark) != -1:
                return  ret
            else:
                line = line.strip()
                idx = line.find(" ")
                k = line[:idx] 
                v = line[idx:].strip()
                # duplicate key
                if k in ret:
                    print("key %s is duplicated!")
                else:
                    ret[k] = v 

def deal_src(src_file):
    with open(src_file, 'r') as f:
        invalid_line_num = 4
        for i in range(invalid_line_num):
            f.readline()
        mark = "-" * 5
        ret = []
        while True:
            line = f.readline()
            line = line.strip()
            if line.find(mark) != -1:
                return ret 
            else:
                ret.append(line.split(" ")[0])


def main():
    mapping_file = "torch_ops_map.txt"
    args = parser.parse_args()
    table = read_src_file(mapping_file)

    src_file = args.src_file
    out_file = args.out_file
    aten_ops = deal_src(src_file)
    not_mapping_ops = []
    zone_line = '-' * 40 + '\n'
    torch_ops = []
    with open(out_file, 'w') as f:    
        f.write("mapping:\n")
        f.write(zone_line)
        for op in aten_ops:
            if not op in table:
                not_mapping_ops.append(op)
            else:
                torch_op =  table[op]
                f.write("%-40s\t%s\n" % (op, torch_op))
                if torch_op != 'x':
                    torch_ops.append(torch_op)
                    
        f.write(zone_line)
        f.write("torch ops:\n")
        f.write(zone_line)

        # delete duplicated op
        torch_ops1 = list(set(torch_ops))
        torch_ops1.sort(key=torch_ops.index)
        torch_ops = torch_ops1

        for torch_op in torch_ops:
            f.write("%s\n" % torch_op)
        f.write(zone_line)
        if len(not_mapping_ops) > 0:
            msg_no_key = "key not exit in table"
            f.write(msg_no_key)
            f.write(zone_line)
            for op in not_mapping_ops:
                f.write("%s\n" % op)
            f.write(zone_line)
            print(msg_no_key)
            print(not_mapping_ops)
        else:
            print("done")
        

if __name__ == "__main__":
    main()

