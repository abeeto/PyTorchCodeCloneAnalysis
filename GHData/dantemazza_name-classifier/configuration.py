import dataParser as data


starting_year = 1990
data_extraction_size = 20000
training_size = 10000
cv_set_size = 2000
test_set_size = 2000


#---gram features---#

di_num = 100
tri_num = 100


#---suffixes---#

last_letters = 8
di_sufnum = 50
tri_sufnum = 50


#---optim variables---#
iterations = 20000
learning_rate = 0.001
minibatch = 100

IS_DEBUG = True