a = [i for i in range(1,12+1)]
print(a)

#逆置列表中的下标为[3,11)的元素
b = a[:3] + a[3:11:-1] + a[11:]
print(b)

c="1a2sd3fg2hjk4laqw2ert2yuiopbiaia21bb0bb54562102bb1b.bb,bbb0a124a511a22a2a242a1a"
d={}

# 统计频率
#d = {i:c.count(i) for i in set(c)}
for i in c: d[i] = d.get(i, 0)+1
print(d)

##对字典按频率排序
d = dict(sorted(d.items(), key=lambda x: x[-1]))
print(d)

#以下第一行为输入，第二行为输出，即求任意多以空格隔开的整数的和
e = sum(list(map(int, input().split())))
print(e)