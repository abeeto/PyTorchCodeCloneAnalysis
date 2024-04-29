import os, sys

task = '3_1' if len(sys.argv) == 1 else sys.argv[1]
print(f'=== Running task {task} ===')
os.system(f'python -m pytest tests/ -m task{task}')

