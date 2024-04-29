import numpy as np

visible_nodes = np.zeros((2, 10, 2))
output_res = 300
x, y = 100, 200
tot = 0
for i in range (2):
  tot = 0
  for j in range(10):
    if x >= 0 and y >= 0  and x < output_res and y < output_res:
                        visible_nodes[i][tot] = (j*output_res**2 + y * output_res + x, 1)
                        tot+=1

print (visible_nodes)

