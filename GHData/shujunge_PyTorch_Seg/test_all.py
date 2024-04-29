from tests.test_datasets import test_testdataset_shape, test_traindataset_shape, test_valdataset_shape, test_traindataset_time

if __name__=="__main__":

    test_traindataset_shape()
    test_valdataset_shape()
    test_testdataset_shape()
    test_traindataset_time()