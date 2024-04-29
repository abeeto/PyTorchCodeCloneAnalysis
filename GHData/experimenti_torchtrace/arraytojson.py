import json
import numpy as np

def jsonFromNumpyArray(ndarray):

	array_dict = {}

	iterations = ndarray.size
	array_dict = dict(np.ndenumerate(ndarray))
		
	return array_dict



