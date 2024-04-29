#!/usr/bin/env python

from time import clock
from numpy import ones

from theano import Mode
from theano import function

from theano.tensor import dscalars
from theano.tensor import dmatrices

from theano.tensor import lt
from theano.tensor import mean

from theano.tensor import switch
from theano.ifelse import ifelse

a_dscalar, b_dscalar = dscalars('a', 'b')
x_dmatrix, y_dmatrix = dmatrices('x', 'y')

z_switch_dmatrix = switch(lt(a_dscalar, b_dscalar), mean(x_dmatrix), mean(y_dmatrix))
z_ifelse_dmatrix = ifelse(lt(a_dscalar, b_dscalar), mean(x_dmatrix), mean(y_dmatrix))

# Both ops build a condition over symbolic variables. IfElse takes a boolean condition and two variables as inputs.
# Switch evaluates both output variables, ifelse is lazy and only evaluates one variable with respect to the condition.
# Unless linker='vm' or linker='cvm' are used, ifelse will compute both variables and take the same computation time as switch.
f_switch = function([a_dscalar, b_dscalar, x_dmatrix, y_dmatrix], z_switch_dmatrix, mode=Mode(linker='vm'))
f_ifelse = function([a_dscalar, b_dscalar, x_dmatrix, y_dmatrix], z_ifelse_dmatrix, mode=Mode(linker='vm'))

var1 = 0.
var2 = 1.
big_mat1 = ones((10000, 1000))
big_mat2 = ones((10000, 1000))

n_times = 10

tic = clock()
for i in xrange(n_times):
    f_switch(var1, var2, big_mat1, big_mat2)
print 'time spent evaluating both values %f sec' % (clock() - tic)

tic = clock()
for i in xrange(n_times):
    f_ifelse(var1, var2, big_mat1, big_mat2) # lazy: only evaluates one variable with respect to the condition
print 'time spent evaluating both values %f sec' % (clock() - tic)


