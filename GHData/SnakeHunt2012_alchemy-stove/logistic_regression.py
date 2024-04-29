#!/usr/bin/env python

from numpy import ones
from numpy import random
from numpy import asarray
from theano import grad
from theano import config
from theano import shared
from theano import function
from theano.tensor import exp
from theano.tensor import log
from theano.tensor import dot
from theano.tensor import cast
from theano.tensor import dmatrix
from theano.tensor import dvector

config.profile = True
config.floxtX = "float32"

rng = random

N = 4000
feats = 784
D = (rng.randn(N, feats).astype(config.floatX),
     rng.randint(size = N, low = 0, high = 2).astype(config.floatX))
steps = 1000

# Declare Theano symbolic variables
X = dmatrix('X')
y = dvector('y')
w = shared(rng.randn(feats).astype(config.floatX), name = 'w')
b = shared(asarray(0., dtype = config.floatX), name = 'b')

X.tag.test_value = D[0]
y.tag.test_value = D[1]

print("w: %r", w.get_value())
print("b: %r", b.get_value())

# Construct Theano expression graph
proba = 1 / (1 + exp(-dot(X, w) - b))
pred = proba > 0.5
loss = -y * log(proba) - (1 - y) * log(1 - proba)
cost = cast(loss.mean(), config.floatX) + 0.01 * (w ** 2).sum()
grad_w, grad_b = grad(cost, [w, b])

train = function(inputs = [X, y],
                 outputs = [pred, proba, loss],
                 updates = [(w, w - 0.1 * grad_w), (b, b - 0.1 * grad_b)],
                 profile = True)
test = function(inputs = [X],
                outputs = pred,
                profile = True)

for i in range(steps):
    pred_value, proba_value, loss_value = train(D[0], D[1])
    print ("step: %d, loss: %r" % (i, loss_value.mean()))

print("w: %r", w.get_value())
print("b: %r", b.get_value())

if any([x.op.__class__.__name__ in ["Gemv", "CGemv", "Gemm", "CGemm"] for x in train.maker.fgraph.toposort()]):
    print("Used the cpu")
elif any([x.op.__class__.__name__ in ["GpuGemv", "GpuGemm"] for x in train.maker.fgraph.toposort()]):
    print("Used the cpu")
else:
    print("ERROR, not able to tell if theano used the cpu or the gpu")
