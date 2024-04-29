import numpy as np
import torch
import time
import argparse
from statistics import mean, stdev

GEN_FILE_HEADER = """
/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_COST
#undef GET_OP_COST

"""

GEN_FILE_TAIL = """

#endif  // GET_OP_COST
"""

BINARY_OPS_TO_MEASURE = [
    (torch.nn.functional.linear, (1, 196), (128, 196)), 
]

def black_box_binary(fn, lhs, rhs, ntimes=10):
    timetraces = []
    for i in range(ntimes):
        start = time.time()
        fn(lhs, rhs)
        end = time.time()
        duration = end - start
        timetraces.append(duration)
    return timetraces

# 12 : sec ; 9 : ms ; 6 : us ; 3 : ns ; 0 : ps
def profile_stats(timetraces):
    # gen statistics, use mean currently, and std
    # maybe need more in future to analysis uncertainty of performances
    return (mean(timetraces), stdev(timetraces))

def tensor_gen(shape):
    print(shape)
    matrix = torch.rand(list(shape)).float();
    return(matrix)

def gen_query_key_binary(operation, lhs_shape, rhs_shape):
    op_str = operation.__name__
    lhs_shape_str = ""
    rhs_shape_str = ""

    for dim in lhs_shape:
        lhs_shape_str += "x{}".format(dim)

    for dim in rhs_shape:
        rhs_shape_str += "x{}".format(dim)

    perf_key = op_str + "-" + lhs_shape_str[1:] + "-" + rhs_shape_str[1:]
    return perf_key

def gen_perf_query_entry_binary(operation, lhs_shape, rhs_shape):
    lhs = tensor_gen(lhs_shape)
    rhs = tensor_gen(rhs_shape)

    # support dispatch and config
    timetraces = black_box_binary(operation, lhs, rhs)
    mean, std = profile_stats(timetraces)
    return (mean, std)

# unit = ms
def gen_value_insert_action(query_key, mean):
    return "this->atomic_cost.insert(std::make_pair(StringRef(\"" + query_key + "\"), " + "{}".format(mean*1000*1000) + "));\n"


if __name__ == '__main__':
    gen_file_str = GEN_FILE_HEADER
    for (operation, lhs_shape, rhs_shape) in BINARY_OPS_TO_MEASURE:
        # print("{}".format(operation))
        query_key = gen_query_key_binary(operation, lhs_shape, rhs_shape)
        # print(query_key)
        mean, _ = gen_perf_query_entry_binary(operation, lhs_shape, rhs_shape)

        gen_cppinc_item = gen_value_insert_action(query_key, mean);
        print(gen_cppinc_item)

        gen_file_str += gen_cppinc_item
    gen_file_str += GEN_FILE_TAIL

    writer = open("CrtOpsCosts.cpp.inc", "w")
    writer.write(gen_file_str)
    writer.close()
   
