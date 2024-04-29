import itertools
class Graph:
    def __init__(self,number_of_people):
        self._number_of_people = number_of_people
        self._dictout = {}
        self._dictin = {}
        self._dictcost = {}
        self._costpeople =[]
        for i in range(0,number_of_people):
            self._costpeople.append(0)
        l = list(itertools.product([0,1],repeat=number_of_people+1))
        for i in range(2**(number_of_people+1)):
            self._dictin[l[i]] = []
            self._dictout[l[i]] = []


    def addEdge(self,origin,destination,cost):
    #   Function that adds a new edge based on given vertices
    #   Pre: x,y - valid vertices and edge(x->y) does not already exist
    #   Returns: -

        self._dictin[destination].append(origin)
        self._dictout[origin].append(destination)

        self._dictcost[(origin,destination)] = cost

    def get_dictin(self):
        return self._dictin

    def get_dictout(self):
        return self._dictout

    def get_dictcost(self):
        return self._dictcost

    def parse_vertices(self):
    #   returns an iterable object containing all vertices in the graph
        return self._dictout.keys()

    def parse_edges(self):
    #   returns an iterable object containing all edges in the graph as a list of tuples
        return self._dictcost.keys()

    def parse_outbound_neighbours(self,x):
    #   Pre: x - valid vertex
    #   Returns an iterable object containing all vertices to which there is an edge from vertex x
        return self._dictout[x]

    def parse_inbound_neighbours(self,x):
    #   Pre: x - valid vertex
    #   Returns an iterable object containing all vertices from which there is an edge to a vertex x
        return self._dictin[x]

    def get_cost(self,x,y):
    #   Function that retrieves the information attached to an edge specified by its vertices
    #   Pre: x,y - valid vertices
    #   Returns: the cost (int) attached to the edge
        return self._dictcost[(x,y)]

    def set_cost(self,x,y,new_cost):
    # Function that modifies the information attached to an edge specified by its vertices
    #Pre: x,y - valid vertices , edge(x->y) is also valid
    #Returns: -
        self._dictcost[(x,y)] = new_cost

    def enter_input(self):
        """
        this function gets the necessary input for the graph
        :return: -
        """
        for i in range(0,self._number_of_people):
            cost = int(input("Enter the time for person "+str(i)+": "))
            self._costpeople[i] = cost

    def check_valid_path(self,source,destination):
        """
        this function checks if the path from source to destination is valid according the the properties of the given problem
        :param source: source vertex
        :param destination: destination vertex
        :return: True - if the path is valid, False - otherwise
        """
        if source[self._number_of_people] == destination[self._number_of_people]:
            return False
        nr = 0
        nr_mistakes =0
        for i in range(0,self._number_of_people):
            if source[i] != destination[i] and destination[i] == destination[self._number_of_people]:
                nr+=1
            elif source[i] != destination[i] and destination[i] != destination[self._number_of_people]:
                nr_mistakes+=1
        if nr >0 and nr < 3 and nr_mistakes == 0:
            return True
        return False

    def find_cost_edge(self,source,destination):
        """
        the function computes the cost of edge according to the people that have crossed the bridge and their time
        :param source: source vertex
        :param destination: destination vertex
        :return: the maximum time of the people
        """
        l=[]
        for i in range(0,self._number_of_people):
            if source[i] != destination[i]:
                l.append(self._costpeople[i])
        return max(l)

    def create_edges(self):
        """
        this function creates the edges of the graph according to the properties given by the problem
        :return: -
        """
        for source in self.parse_vertices():
            for destination in self.parse_vertices():
                if self.check_valid_path(source,destination):
                    self._dictcost[(source,destination)] = self.find_cost_edge(source,destination)
                    self._dictout[source].append(destination)
                    self._dictin[destination].append(source)


class PriorityQueue:
    def __init__(self):
        self._values = {}

    def isEmpty(self):
        """
        this function checks if the priority queue is empty
        :return: True - if it is empty, False, otherwise
        """
        return len(self._values) == 0

    def pop(self):
        """
        this function pops the value from the priority queue with the top priority
        :return: the topObject
        """
        topPriority = None
        topObject = None
        for obj in self._values:
            objPriority = self._values[obj]
            if topPriority is None or topPriority > objPriority:
                topPriority = objPriority
                topObject = obj
        del self._values[topObject]
        return topObject

    def add(self,obj,priority):
        """
        this function adds an object to the priority queue
        :param obj: the object addes
        :param priority: the priority of the object
        :return: -
        """
        self._values[obj]=priority

    def contains(self,val):
        """
        checks if the value "val" is in the queue or not
        :param val: the value
        :return: True if it is in the queue, False - otherwise
        """
        return val in self._values


def getChildren(x, prev):
    list = []
    for i in prev :
        if prev[i] == x:
            list.append(i)
    return list


def printDijkstraTree(s, q, d, prev, indent):
    if q.contains(s):
        star=''
    else:
        star='*'
    print ("%s%s [%s]%s " % (indent, s, d[s], star))
    for x in getChildren(s,prev):
        printDijkstraTree(x, q, d, prev, indent+'    ')


def printDijkstraStep(s, x, q, d, prev):
    print ('-----')
    if x is not None:
        print ('x=%s [%s]' % (x,d[x]))
    printDijkstraTree(s,q,d,prev,'')


def dijkstra(g, s, destination):
    """
    this function computes the lowest cost path from s to destianation using Dijkstra algorithm
    :param g: the graph
    :param s: source vertex
    :param destination: destination vertex
    :return: the list of distances d , the dictionary of previous vertices "prev"
    """
    prev = {}
    q = PriorityQueue()
    q.add(s,0)
    d = {}
    d[s]=0
    visited = set()
    visited.add(s)
    #printDijkstraStep(s,None,q,d,prev)
    while not q.isEmpty():
        x = q.pop()
        for y in g.parse_outbound_neighbours(x):
            if y not in visited or d[y] > d[x] + g.get_cost(x,y):
                d[y] = d[x] + g.get_cost(x,y)
                visited.add(y)
                q.add(y,d[y])
                prev[y] = x
        #printDijkstraStep(s, x, q, d, prev)
        if x == destination :
            return (d,prev)
    return (d,prev)


def getPath(s, t, prev):
    """
    this function computes the path from s to t using the "prev" dictionary computed in Dijktra algorithm
    :param s: source vertex
    :param t: destination vertex
    :param prev: dictionary containing the previous vertex of every vertex
    :return: the path
    """
    list = []
    while t != s:
        list.append(t)
        t = prev[t]
    ret = [s]
    for i in range(len(list)):
        ret.append(list[len(list) - i - 1])
    return ret


source = ()
destination = ()
number_of_people = int(input("Enter the number of people to cross the bridge: "))
g = Graph(number_of_people)
g.enter_input()
g.create_edges()
for i in range(0, number_of_people+1):
    source = source+(0,)
    destination=destination+(1,)
d, prev = dijkstra(g, source, destination)
print(getPath(source, destination, prev))
print(d[destination])
print("//1 means it crossed the bridge")
