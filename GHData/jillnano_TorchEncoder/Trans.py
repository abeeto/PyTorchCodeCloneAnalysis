# !usr/bin/python
# coding=utf-8

import json

content = open('td.txt', 'r').readlines()
result = {}
for i in content:
	name = i.split('@|@')[0].split('/Music/')[-1]
	v = json.loads(i.split('@|@')[-1])
	value = []
	for t in v:
		value.append(json.loads(t))
	result[name] = value

f = open('data.json', 'w')
f.write(json.dumps(result))
f.close()