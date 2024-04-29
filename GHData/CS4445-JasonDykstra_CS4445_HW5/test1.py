from problem1 import *
import sys
import math
from gradient import *
import warnings
'''
    Unit test 1:
    This file includes unit tests for problem1.py.
'''

#-------------------------------------------------------------------------
def test_python_version():
    ''' ----------- Problem 1 (60 points in total)---------------------'''
    assert sys.version_info[0]==3 # require python 3.6 or above 
    assert sys.version_info[1]>=6

#---------------------------------------------------
def test_compute_z():
    ''' (5 points) compute_z'''
    x = np.array([1., 2.])
    w = np.array([0.5, -0.6])
    b = 0.2
    z = compute_z(x,w,b)
    assert np.allclose(z, -0.5, atol = 1e-3) 
    w = np.array([-0.5, 0.6])
    z = compute_z(x,w,b)
    assert np.allclose(z, .9, atol = 1e-3) 
    w = np.array([0.5,-0.6])
    x = np.array([ 2., 5. ])
    z = compute_z(x,w,b)
    assert np.allclose(z, -1.8, atol = 1e-3) 
    b = 0.5
    z = compute_z(x,w,b)
    assert np.allclose(z, -1.5, atol = 1e-3) 
#---------------------------------------------------
def test_compute_dz_db():
    ''' (5 points) compute_dz_db'''
    for _ in range(20):
        p = np.random.randint(2,20)
        x = np.random.random(p)
        w = np.random.random(p)
        b = np.random.random(1)
        # analytical gradients
        db = compute_dz_db()
        # numerical gradients
        db_true = check_dz_db(x,w,b)
        assert np.allclose(db, db_true, atol=1e-2) 
#---------------------------------------------------
def test_compute_dz_dw():
    ''' (5 points) compute_dz_dw'''
    for _ in range(20):
        p = np.random.randint(2,20)
        x = 2*np.random.random(p)-1
        w = 2*np.random.random(p)-1
        b = 2*np.random.random(1)[0]-1
        # analytical gradients
        dw = compute_dz_dw(x)
        # numerical gradients
        dw_true = check_dz_dw(x,w,b)
        assert np.allclose(dw, dw_true, atol=1e-2) 
#---------------------------------------------------
def test_compute_L():
    ''' (5 points) compute_L'''
    L= compute_L(0.,0)
    assert np.allclose(L, np.log(2), atol = 1e-3) 
    L= compute_L(0.,1)
    assert np.allclose(L, np.log(2), atol = 1e-3) 
    warnings.filterwarnings("error")
    L= compute_L(1000.,0)
    assert np.allclose(L, 1000., atol = 1e-1) 
    L= compute_L(2000.,0)
    assert np.allclose(L, 2000., atol = 1e-1) 
    L= compute_L(1000.,1)
    assert np.allclose(L, 0., atol = 1e-1) 
    L= compute_L(2000.,1)
    assert np.allclose(L, 0., atol = 1e-1) 
    L= compute_L(-1000.,0)
    assert np.allclose(L, 0., atol = 1e-1) 
    L= compute_L(-2000.,0)
    assert np.allclose(L, 0., atol = 1e-1) 
    L= compute_L(-1000.,1)
    assert np.allclose(L, 1000., atol = 1e-1) 
    L= compute_L(-2000.,1)
    assert np.allclose(L, 2000., atol = 1e-1) 
#---------------------------------------------------
def test_compute_dL_dz():
    ''' (5 points) compute_dL_dz'''
    dL_dz = compute_dL_dz(0,0)
    assert np.allclose(dL_dz, 0.5, atol= 1e-3)
    dL_dz = compute_dL_dz(0,1)
    assert np.allclose(dL_dz, -0.5, atol= 1e-3)
    dL_dz = compute_dL_dz(1000,1)
    assert dL_dz == dL_dz # check if dL_dz is NaN (not a number)
    assert np.allclose(dL_dz, 0., atol= 1e-3)
    dL_dz = compute_dL_dz(1000,0)
    assert dL_dz == dL_dz # check if dL_dz is NaN (not a number)
    assert np.allclose(dL_dz, 1., atol= 1e-3)
    warnings.filterwarnings("error")
    dL_dz = compute_dL_dz(-1000,0)
    assert np.allclose(dL_dz, 0., atol= 1e-3)
    dL_dz = compute_dL_dz(-1000,1)
    assert np.allclose(dL_dz, -1., atol= 1e-3)
    for _ in range(20):
        z = 10*np.random.random(1)[0]-5
        y = np.random.randint(2)
        # analytical gradients
        dz = compute_dL_dz(z,y)
        # numerical gradients
        dz_true = check_dL_dz(z,y)
        assert np.allclose(dz, dz_true, atol=1e-2) 
#---------------------------------------------------
def test_compute_dL_db():
    ''' (5 points) compute_dL_db'''
    dL_dz = -2.0 
    dz_db = 1.0 
    dL_db = compute_dL_db(dL_dz,dz_db)
    dL_db_true = -2.0
    assert np.allclose(dL_db, dL_db_true, atol = 1e-3)
#---------------------------------------------------
def test_compute_dL_dw():
    ''' (5 points) compute_dL_dw'''
    dL_dz = -1.0
    dz_dw = np.array([1., 2.])
    dL_dw = compute_dL_dw(dL_dz, dz_dw)
    dL_dw_true =np.array([-1., -2.])
    assert np.allclose(dL_dw, dL_dw_true, atol = 1e-3)
    dL_dz = 0.5
    dz_dw = np.array([2., 3.])
    dL_dw = compute_dL_dw(dL_dz, dz_dw)
    dL_dw_true =np.array([1., 1.5])
    assert np.allclose(dL_dw, dL_dw_true, atol = 1e-3)
#---------------------------------------------------
def test_backward():
    ''' (5 points) backward'''
    warnings.filterwarnings("error")
    x = np.array([1., 2.])
    y = 1 
    z = 0
    dL_dw, dL_db = backward(x,y,z)
    assert np.allclose(dL_dw,[-0.5,-1], atol=1e-3)
    assert np.allclose(dL_db,-0.5, atol=1e-3)
    x = np.array([2., 3., 4.])
    y = 1
    z = 1000.
    dL_dw, dL_db = backward(x,y,z)
    assert np.allclose(dL_dw,[0,0,0], atol=1e-3)
    assert np.allclose(dL_db,0, atol=1e-3)
    y = 1
    z = -1000.
    dL_dw, dL_db = backward(x,y,z)
    assert np.allclose(dL_dw,[-2,-3,-4], atol=1e-3)
    assert np.allclose(dL_db,-1, atol=1e-3)
    y = 0
    z = -1000.
    dL_dw, dL_db = backward(x,y,z)
    assert np.allclose(dL_dw,[0,0,0], atol=1e-3)
    assert np.allclose(dL_db,0, atol=1e-3)
    y = 0
    z = 1000.
    dL_dw, dL_db = backward(x,y,z)
    assert np.allclose(dL_dw,[2,3,4], atol=1e-3)
    assert np.allclose(dL_db,1, atol=1e-3)
#---------------------------------------------------
def test_update_b():
    ''' (5 points) update_b'''
    b = 0.
    dL_db = 2.
    b = update_b(b, dL_db, alpha=.5) 
    b_true = -1.
    assert np.allclose(b, b_true, atol = 1e-3)
    b = update_b(b, dL_db, alpha=1.)
    b_true = -3.
    assert np.allclose(b, b_true, atol = 1e-3)
#---------------------------------------------------
def test_update_w():
    ''' (5 points) update_w'''
    w = np.array( [0., 0.])
    dL_dw = np.array( [1., 2.])
    w = update_w(w,dL_dw, alpha=.5) 
    w_true = - np.array([0.5, 1.])
    assert np.allclose(w, w_true, atol = 1e-3)
    w = update_w(w,dL_dw, alpha=1.) 
    w_true = - np.array([1.5, 3.])
    assert np.allclose(w, w_true, atol = 1e-3)
#---------------------------------------------------
def test_train():
    ''' (5 points) train'''
    X = np.array([[0., 1.], # an example feature matrix (4 instances, 2 features)
                  [1., 0.],
                  [0., 0.],
                  [1., 1.]])
    Y = np.array([0, 1, 0, 1])
    w, b = train(X, Y, alpha=1., n_epoch = 100)
    assert w[1] + b <= 0 # x1 is negative 
    assert w[0] + b >= 0 # x2 is positive
    assert  b <= 0 # x3 is negative 
    assert w[0]+w[1] + b >= 0 # x4 is positive
    X = np.array([[0., 1.],
                  [1., 0.],
                  [0., 0.],
                  [2., 0.],
                  [0., 2.],
                  [1., 1.]])
    Y = np.array([0, 0, 0, 1, 1, 1])
    w, b = train(X, Y, alpha=0.1, n_epoch = 1000)
    assert w[0]+w[1] + b >= 0 
    assert 2*w[0] + b >= 0 
    assert 2*w[1] + b >= 0 
    assert w[0] + b <= 0 
    assert w[1] + b <= 0 
    assert  b <= 0 
#---------------------------------------------------
def test_predict():
    ''' (5 points) predict'''
    x= np.array([1,1])
    w = np.array([ 0.5, -0.6])
    b = 0.2
    y = predict(x, w, b)
    assert y==1
    x= np.array([0,1])
    y= predict(x, w, b )
    assert y==0
    x= np.array([2,2])
    y= predict(x, w, b )
    assert y==1

