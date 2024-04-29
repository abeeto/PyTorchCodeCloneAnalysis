import unittest
import arraytojson
import numpy as np
import harperdb as hdb
import checksize

class TestDynamicArrayToJson(unittest.TestCase):

    def test_dir_size(self):
        result = checksize.get_size('/data/hdb')
        print ("Directory Size")
        print (result)
        self.assertTrue(result)

    def test_ping(self):
        result = hdb.ping()
        self.assertTrue(result)

    def test_dimension_scale(self):
        #1000x100 matrix
        w1 = np.random.randn(1000, 100)
        result = arraytojson.jsonFromNumpyArray(w1)
        print(w1.shape)
       # self.assertTrue(len(result)==999)

    def test_enumerate_narray(self):
        w1 = np.random.randn(100, 100)
        for k, v in np.ndenumerate(w1):
             pass
             #print('k ', k)
             #print('v ', v)


if __name__ == '__main__':
    unittest.main()