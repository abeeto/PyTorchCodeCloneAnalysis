data = list(map(int, input().split(',')))
print(data)
max_data, min_max = data[0], data[0]

for i in range(1, len(data)):
    if max_data < data[i]:
        min_data = max_data
        max_data = data[i]
print(min_data)