import re

real_time={}
real_mem={}
trace_mem={}
trace_time={}
pattern = re.compile(r'^Name:[ ]*(\w+) Batch:[ ]*(\d+) Eval:[ ]*([.0-9]+)ms Trace:[ ]*([.0-9]+)ms Eval Cuda Size:[ ]*(\d+)KB Traced Cuda Size:[ ]*(\d+)KB$')
with open("res.txt","r") as res:
    lines = res.readlines()
    for line in lines:
        matcher = pattern.match(line.strip('\n'))
        if matcher.group(1) not in real_time:
            real_time[matcher.group(1)] = {}
        if matcher.group(1) not in real_mem:
            real_mem[matcher.group(1)] = {}
        if matcher.group(1) not in trace_time:
            trace_time[matcher.group(1)] = {}
        if matcher.group(1) not in trace_mem:
            trace_mem[matcher.group(1)] = {}
        real_time[matcher.group(1)][matcher.group(2)] = matcher.group(3)
        trace_time[matcher.group(1)][matcher.group(2)] = matcher.group(4)
        real_mem[matcher.group(1)][matcher.group(2)] = matcher.group(5)
        trace_mem[matcher.group(1)][matcher.group(2)] = matcher.group(6)
print("Name\t1\t2\t4\t8\t16\t1\t2\t4\t8\t16\t1\t2\t4\t8\t16\t1\t2\t4\t8\t16\t")
for key,value in real_time.items():
    print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t".format(key,
        real_time[key]["1"], real_time[key]["2"], real_time[key]["4"], real_time[key]["8"], real_time[key]["16"],
        real_mem[key]["1"], real_mem[key]["2"], real_mem[key]["4"], real_mem[key]["8"], real_mem[key]["16"],
        trace_time[key]["1"], trace_time[key]["2"], trace_time[key]["4"], trace_time[key]["8"], trace_time[key]["16"],
        trace_mem[key]["1"], trace_mem[key]["2"], trace_mem[key]["4"], trace_mem[key]["8"], trace_mem[key]["16"]
        ))
