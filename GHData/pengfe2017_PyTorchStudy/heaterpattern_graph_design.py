graph = {
  'A' : ['B','C'],
  'B' : ['D', 'E'],
  'C' : ['F'],
  'D' : [],
  'E' : ['F'],
  'F' : []
}

visited = [] # List to keep track of visited nodes.
queue = []     #Initialize a queue

#def bfs(visited, graph, node):
visited.append(node)
queue.append(node)

while queue:
s = queue.pop(0) 
print (s, end = " ") 

for neighbor in graph[s]:
    if naghbor not in visited:
    visited.append(naghbor)
    queue.append(naghbor)



# Driver Code
#bfs(visited, graph, 'A')