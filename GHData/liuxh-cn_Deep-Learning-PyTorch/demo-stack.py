import numpy as np

z1 = np.array(([1,2,3]))
z2 = np.array((4,5,6))
z3 = np.array((11,22,33))
z4 = np.array((44,55,66))
t = np.vstack((z1,z2))
t = np.hstack((z3, z4))
t = np.hstack(([z1, z2], [z3, z4]))