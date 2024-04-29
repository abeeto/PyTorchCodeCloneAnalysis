from src.optimTester import optimTester, pframe

tester = optimTester(pframe)

init = 10
end = 100000
step = end//15

for i, iters in enumerate(range(init, end, step)):
    solverParams = dict(jit=True,maxiter=iters,tol=1e-6)
    tester.testOptim(solverParams, _iter=i, maxiter=iters)
