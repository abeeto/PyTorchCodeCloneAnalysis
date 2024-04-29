import numpy as np
from functools import reduce
def main():
    """
    This is sample code for linear regression, which demonstrates how to use the
    Graph class.

    
    """

    # This is our data, where x is a 4x2 matrix and y is a 4x1 matrix
    x = np.array([[0., 0.],
                  [0., 1.],
                  [1., 0.],
                  [1., 1.]])
    y = np.dot(x, np.array([[7.],
                            [8.]])) + 3

    # Let's construct a simple model to approximate a function from 2D
    # points to numbers, f(x) = x_0 * m_0 + x_1 * m_1 + b
    # Here m and b are variables (trainable parameters):
    m = Variable(2,1)
    b = Variable(1)

    # We train our network using batch gradient descent on our data
    for iteration in range(10000):
        # At each iteration, we first calculate a loss that measures how
        # good our network is. The graph keeps track of all operations used
        graph = Graph([m, b])
        input_x = Input(graph, x)
        input_y = Input(graph, y)
        xm = MatrixMultiply(graph, input_x, m)
        xm_plus_b = MatrixVectorAdd(graph, xm, b)
        loss = SquareLoss(graph, xm_plus_b, input_y)
        # Then we use the graph to perform backprop and update our variables
        graph.backprop()
        graph.step(0.01)

    # After training, we should have recovered m=[[7],[8]] and b=[3]
    print("Final values are: {}".format([m.data[0,0], m.data[1,0], b.data[0]]))
    assert np.isclose(m.data[0,0], 7)
    assert np.isclose(m.data[1,0], 8)
    assert np.isclose(b.data[0], 3)
    print("Success!")

class Graph(object):
    """
    A graph that keeps track of the computations performed by a neural network
    in order to implement back-propagation.

    Each evaluation of the neural network (during both training and test-time)
    will create a new Graph. The computation will add nodes to the graph, where
    each node is either a DataNode or a FunctionNode.

    A DataNode represents a trainable parameter or an input to the computation.
    A FunctionNode represents doing a computation based on two previous nodes in
    the graph.

    The Graph is responsible for keeping track of all nodes and the order they
    are added to the graph, for computing gradients using back-propagation, and
    for performing updates to the trainable parameters.

    For an example of how the Graph can be used, see the function `main` above.
    """

    def __init__(self, variables):
        """
        Initializes a new computation graph.

        variables: a list of Variable objects that store the trainable parameters
            for the neural network.

        Hint: each Variable is also a node that needs to be added to the graph,
        so don't forget to call `self.add` on each of the variables.
        """
        self.nodes=[]
        self.donebackprop=False
        self.backprops={}
        self.forwardprops={}
        self.gradients={}
        self.variables=variables
        for i in variables:
            self.add(i)

    def get_nodes(self):
        """
        Returns a list of all nodes that have been added to this Graph, in the
        order they were added. This list should include all of the Variable
        nodes that were passed to `Graph.__init__`.

        Returns: a list of nodes
        """
        return self.nodes

    def get_inputs(self, node):
        """
        Retrieves the inputs to a node in the graph. Assume the `node` has
        already been added to the graph.

        Returns: a list of numpy arrays

        Hint: every node has a `.get_parents()` method
        """
        return [self.get_output(parent) for parent in node.get_parents()]

    def get_output(self, node):
        """
        Retrieves the output to a node in the graph. Assume the `node` has
        already been added to the graph.

        Returns: a numpy array or a scalar
        """
        return self.forwardprops[node]

    def get_gradient(self, node):
        """
        Retrieves the gradient for a node in the graph. Assume the `node` has
        already been added to the graph.

        If `Graph.backprop` has already been called, this should return the
        gradient of the loss with respect to the output of the node. If
        `Graph.backprop` has not been called, it should instead return a numpy
        array with correct shape to hold the gradient, but with all entries set
        to zero.

        Returns: a numpy array
        """

        # if self.donebackprop:
        #     print(self.backprops[node])
        #     return self.backprops[node]
        # else:
        #     x=np.zeros_like(self.get_output((node)))
        #     #self.backprops[node]=x
        #     return x
        if self.donebackprop:
            return self.gradients[node]
        else:
            x = np.zeros_like(self.get_output((node)))

            return x
    def add(self, node):
        """
        Adds a node to the graph.

        This method should calculate and remember the output of the node in the
        forwards pass (which can later be retrieved by calling `get_output`)
        We compute the output here because we only want to compute it once,
        whereas we may wish to call `get_output` multiple times.

        Additionally, this method should initialize an all-zero gradient
        accumulator for the node, with correct shape.
        """
        self.nodes+=[node]
        self.forwardprops[node] = node.forward(self.get_inputs(node))
        x=np.zeros_like(self.get_output((node)))
        self.gradients[node]=x




    def backprop(self):
        """
        Runs back-propagation. Assume that the very last node added to the graph
        represents the loss.

        After back-propagation completes, `get_gradient(node)` should return the
        gradient of the loss with respect to the `node`.

        Hint: the gradient of the loss with respect to itself is 1.0, and
        back-propagation should process nodes in the exact opposite of the order
        in which they were added to the graph.
        """
        loss_node = self.get_nodes()[-1]
        reverseorder=reversed(self.get_nodes())
        assert np.asarray(self.get_output(loss_node)).ndim == 0
        self.gradients[loss_node]=1.0






        for node in reverseorder:
            back=node.backward(self.get_inputs(node), self.gradients[node])
            for i in range(len(node.get_parents())):
                child=node.get_parents()[i]
                if child not in self.gradients.keys():

                    self.gradients[child]=back[i]
                else:
                    #print(back[i])
                    x=back[i]
                    self.gradients[child] =np.add(self.gradients[child],x)









        # for node in reverseorder:
        #     children=[child for child in self.nodes if node in child.get_parents()]
        #     gradientsy=[self.backprops[child] for child in children]
        #     gradient= reduce(lambda x, y: (x + y) , gradientsy)
        #     back=node.backward(self.get_inputs(node),gradient)
        #     if back==[]:
        #         pass
        #     else:
        #         self.backprops[node]=back
        self.donebackprop=True

    def step(self, step_size):
        """
        Updates the values of all variables based on computed gradients.
        Assume that `backprop()` has already been called, and that gradients
        have already been computed.

        Hint: each Variable has a `.data` attribute
        """
        y=self.get_output(self.get_nodes()[-1])
        #print(y)

        for variable in self.variables:
            #print(step_size*self.gradients[variable])
            variable.data-=step_size*self.gradients[variable]
        #print([variable.data for variable in self.variables])


class DataNode(object):
    """
    DataNode is the parent class for Variable and Input nodes.

    Each DataNode must define a `.data` attribute, which represents the data
    stored at the node.
    """

    @staticmethod
    def get_parents():
        # A DataNode has no parent nodes, only a `.data` attribute
        return []

    def forward(self, inputs):
        # The forwards pass for a data node simply returns its data
        return self.data

    @staticmethod
    def backward(inputs, gradient):
        # A DataNode has no parents or inputs, so there are no gradients to
        # compute in the backwards pass
        return []

class Variable(DataNode):
    """
    A Variable stores parameters used in a neural network.

    Variables should be created once and then passed to all future Graph
    constructors. Use `.data` to access or modify the numpy array of parameters.
    """

    def __init__(self, *shape):
        """
        Initializes a Variable with a given shape.

        For example, Variable(5) will create 5-dimensional vector variable,
        while Variable(10, 10) will create a 10x10 matrix variable.

        The initial value of the variable before training starts can have a big
        effect on how long the network takes to train. The provided initializer
        works well across a wide range of applications.
        """
        assert shape
        limit = np.sqrt(3.0 / np.mean(shape))
        self.data = np.random.uniform(low=-limit, high=limit, size=shape)

class Input(DataNode):
    """
    An Input node packages a numpy array into a node in a computation graph.
    Use this node for inputs to your neural network.

    For trainable parameters, use Variable instead.
    """

    def __init__(self, graph, data):
        """
        Initializes a new Input and adds it to a graph.
        """
        assert isinstance(data, np.ndarray), "data must be a numpy array"
        assert data.dtype.kind == "f", "data must have floating-point entries"
        self.data = data
        graph.add(self)

class FunctionNode(object):
    """
    A FunctionNode represents a value that is computed based on other nodes in
    the graph. Each function must implement both a forward and backward pass.
    """

    def __init__(self, graph, *parents):
        self.parents = parents
        graph.add(self)

    def get_parents(self):
        return self.parents

    @staticmethod
    def forward(inputs):
        raise NotImplementedError

    @staticmethod
    def backward(inputs, gradient):
        raise NotImplementedError

class Add(FunctionNode):
    """
    Adds two vectors or matrices, element-wise

    Inputs: [x, y]
        x may represent either a vector or a matrix
        y must have the same shape as x
    Output: x + y
    """

    @staticmethod
    def forward(inputs):
        # return sum(inputs)
        return inputs[0]+inputs[1]
    @staticmethod
    def backward(inputs, gradient):
        return [gradient*np.ones(inputs[0].shape),gradient*np.ones(inputs[1].shape)]
        # return [1 for i in len(inputs)]

class MatrixMultiply(FunctionNode):
    """
    Represents matrix multiplication.

    Inputs: [A, B]
        A represents a matrix of shape (n x m)
        B represents a matrix of shape (m x k)
    Output: a matrix of shape (n x k)
    """

    @staticmethod
    def forward(inputs):
        #return reduce(lambda x, y: np.dot(x,y), inputs)
        return np.dot(inputs[0],inputs[1])

    @staticmethod
    def backward(inputs, gradient):
        # partial_prods=forward(inputs)
        # return [ for input in inputs]
        return [np.dot(gradient,inputs[1].T),np.dot(inputs[0].T,gradient)]

class MatrixVectorAdd(FunctionNode):
    """
    Adds a vector to each row of a matrix.

    Inputs: [A, x]
        A represents a matrix of shape (n x m)
        x represents a vector (m)
    Output: a matrix of shape (n x m)
    """

    @staticmethod
    def forward(inputs):
        return inputs[0]+inputs[1]

    @staticmethod
    def backward(inputs, gradient):
        n,m=inputs[0].shape
        return [gradient*np.ones((n,m)),np.sum((gradient*np.ones(m)),axis=0)]

class ReLU(FunctionNode):
    """
    An element-wise Rectified Linear Unit nonlinearity: max(x, 0).
    This nonlinearity replaces all negative entries in its input with zeros.

    Input: [x]
        x represents either a vector or matrix
    Output: same shape as x, with no negative entries
    """

    @staticmethod
    def forward(inputs):
        return np.maximum(inputs[0],np.zeros(inputs[0].shape))

    @staticmethod
    def backward(inputs, gradient):
        return [gradient*np.where(inputs[0]>0,np.ones(inputs[0].shape),np.zeros(inputs[0].shape))]

class SquareLoss(FunctionNode):
    """
    Inputs: [a, b]
        a represents a matrix of size (batch_size x dim)
        b must have the same shape as a
    Output: a number

    This node first computes 0.5 * (a[i,j] - b[i,j])**2 at all positions (i,j)
    in the inputs, which creates a (batch_size x dim) matrix. It then calculates
    and returns the mean of all elements in this matrix.
    """

    @staticmethod
    def forward(inputs):
        comp=.5*(inputs[0]-inputs[1])**2
        return np.mean(comp)

    @staticmethod
    def backward(inputs, gradient):
        #return [np.dot(.5*gradient,(inputs[0]-inputs[1])),(np.dot(.5*gradient,(inputs[1]-inputs[0]).T)).T]


        # print(inputs[0] - inputs[1])
        # print(1/)
        # print(gradient)
        x=np.dot(1.0/(inputs[0].shape[0]*inputs[0].shape[1])*gradient,inputs[0]-inputs[1])
        #print(x)
        y=np.dot(1.0/(inputs[0].shape[0]*inputs[0].shape[1])* gradient,(inputs[1]-inputs[0]))
        #print(y)
        return [x,y]

class SoftmaxLoss(FunctionNode):
    """
    A batched softmax loss, used for classification problems.

    IMPORTANT: do not swap the order of the inputs to this node!

    Inputs: [logits, labels]
        logits: a (batch_size x num_classes) matrix of scores, that is typically
            calculated based on previous layers. Each score can be an arbitrary
            real number.
        labels: a (batch_size x num_classes) matrix that encodes the correct
            labels for the examples. All entries must be non-negative and the
            sum of values along each row should be 1.
    Output: a number

    We have provided the complete implementation for your convenience.
    """
    @staticmethod
    def softmax(input):
        exp = np.exp(input - np.max(input, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    @staticmethod
    def forward(inputs):
        softmax = SoftmaxLoss.softmax(inputs[0])
        labels = inputs[1]
        assert np.all(labels >= 0), \
            "Labels input to SoftmaxLoss must be non-negative. (Did you pass the inputs in the right order?)"
        assert np.allclose(np.sum(labels, axis=1), np.ones(labels.shape[0])), \
            "Labels input to SoftmaxLoss do not sum to 1 along each row. (Did you pass the inputs in the right order?)"

        return np.mean(-np.sum(labels * np.log(softmax), axis=1))

    @staticmethod
    def backward(inputs, gradient):
        softmax = SoftmaxLoss.softmax(inputs[0])
        return [
            gradient * (softmax - inputs[1]) / inputs[0].shape[0],
            gradient * (-np.log(softmax)) / inputs[0].shape[0]
        ]

if __name__ == '__main__':
    main()
