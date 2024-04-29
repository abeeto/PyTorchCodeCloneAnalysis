




import pickle
infile = open("model_run_1.pk", 'rb')
run_1_0, run_1_1 = pickle.load(infile)
infile.close()

infile = open("model_run_2.pk", 'rb')
run_2_0, run_2_1 = pickle.load(infile)
infile.close()

import numpy as np

for (a_name, a_vals), (b_name, b_vals) in zip(run_1_0.items(), run_2_0.items()):
    np_a_vals = np.array(a_vals)
    np_b_vals = np.array(b_vals)
    #print(np.testing.assert_allclose(np_a_vals, np_b_vals, rtol=1e-4))
    print(a_name, np.abs(np_a_vals-np_b_vals).max())

for (a_name, a_vals), (b_name, b_vals) in zip(run_1_1.items(), run_2_1.items()):
    np_a_vals = np.array(a_vals)
    np_b_vals = np.array(b_vals)
    #print(np.testing.assert_allclose(np_a_vals, np_b_vals, rtol=1e-4))
    print(a_name, np.abs(np_a_vals-np_b_vals).max())
