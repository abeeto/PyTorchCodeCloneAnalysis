import itertools
l_one = [[0,0,1],[1,1,2,3],[2,4,5]]
print(l_one)
print(list(itertools.zip_longest(*l_one,fillvalue=0)))

