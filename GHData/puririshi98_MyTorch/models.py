import numpy as np

import backend
import nn

class Model(object):
    """Base model class for the different applications"""
    def __init__(self):
        self.get_data_and_monitor = None
        self.learning_rate = 0.0

    def run(self, x, y=None):
        raise NotImplementedError("Model.run must be overriden by subclasses")

    def train(self):
        """
        Train the model.

        `get_data_and_monitor` will yield data points one at a time. In between
        yielding data points, it will also monitor performance, draw graphics,
        and assist with automated grading. The model (self) is passed as an
        argument to `get_data_and_monitor`, which allows the monitoring code to
        evaluate the model on examples from the validation set.
        """
        for x, y in self.get_data_and_monitor(self):
            graph = self.run(x, y)
            graph.backprop()

            graph.step(self.learning_rate)
            #print([variable.data for variable in graph.variables])

class RegressionModel(Model):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. 
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_regression

        
        self.learning_rate=.02
        self.hidden_size=200
        self.batch_size=200
        self.graphy=None
        self.W1, self.W2 ,self.b1,self.b2= nn.Variable(1,self.hidden_size), nn.Variable(self.hidden_size,1),nn.Variable(self.hidden_size),nn.Variable(1)

    def run(self, x, y=None):
        """
        Runs the model for a batch of examples.

        The correct outputs `y` are known during training, but not at test time.
        If correct outputs `y` are provided, this method must construct and
        return a nn.Graph for computing the training loss. If `y` is None, this
        method must instead return predicted y-values.

        Inputs:
            x: a (batch_size x 1) numpy array
            y: a (batch_size x 1) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 1) numpy array of predicted y-values

        Note: DO NOT call backprop() or step() inside this method!
        """
        

        if y is not None:
            # At training time, the correct output `y` is known.
            # Here, you should construct a loss node, and return the nn.Graph
            # that the node belongs to. The loss node must be the last node
            # added to the graph.

            graph=nn.Graph([self.W1,self.W2,self.b1,self.b2])
            input_x=nn.Input(graph, x)
            input_y = nn.Input(graph, y)
            xdotW=nn.MatrixMultiply(graph, input_x, self.W1)
            xWplusb1=nn.MatrixVectorAdd(graph,xdotW,self.b1)
            relu_xdotW=nn.ReLU(graph,xWplusb1)
            #print(graph.get_output((relu_xdotW)).shape)
            second_layer=nn.MatrixMultiply(graph,relu_xdotW,self.W2)
            second_layerplusb=nn.MatrixVectorAdd(graph,second_layer,self.b2)
            #print(graph.get_output((second_layer)).shape)
            loss=nn.SquareLoss(graph,second_layerplusb,input_y)
            self.graphy=graph
            return graph
        else:
            # At test time, the correct output is unknown.
            # You should instead return your model's prediction as a numpy array
            #print(self.graphy.get_output(self.graphy.get_nodes()[-2]))
            return self.graphy.get_output(self.graphy.get_nodes()[-2])

class OddRegressionModel(Model):
    """
    A neural network model for approximating an odd function that maps from real
    numbers to real numbers.

    
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_regression

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        self.learning_rate = .005
        self.hidden_size = 200
        self.batch_size = 16
        self.graphy=None
        self.W1, self.W2, self.b1, self.b2 = nn.Variable(1, self.hidden_size), nn.Variable(self.hidden_size,1), nn.Variable(self.hidden_size), nn.Variable(1)
        # self.W1pos, self.W2pos, self.b1pos,self.b2pos = nn.Variable(1, self.hidden_size), nn.Variable(self.hidden_size,1), nn.Variable(self.hidden_size), nn.Variable(1),
        # self.W1neg, self.W2neg, self.b1neg, self.b2neg= nn.Variable(1, self.hidden_size), nn.Variable(self.hidden_size,1), nn.Variable(self.hidden_size), nn.Variable(1)
        # self.W1neg.data, self.W2neg.data, self.b1neg.data, self.b2neg.data=self.W1pos.data, self.W2pos.data, self.b1pos.data,self.b2pos.data

    def run(self, x, y=None):
        """
        Runs the model for a batch of examples.

        The correct outputs `y` are known during training, but not at test time.
        If correct outputs `y` are provided, this method must construct and
        return a nn.Graph for computing the training loss. If `y` is None, this
        method must instead return predicted y-values.

        Inputs:
            x: a (batch_size x 1) numpy array
            y: a (batch_size x 1) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 1) numpy array of predicted y-values

        Note: DO NOT call backprop() or step() inside this method!
        """
        "*** YOUR CODE HERE ***"
        # graph = nn.Graph([self.W1, self.W2, self.b1, self.b2])
        # input_x = nn.Input(graph, x)
        # input_negx = nn.Input(graph, np.dot(np.array(-1.0), x))
        #
        # xdotWpos = nn.MatrixMultiply(graph, input_x, self.W1)
        # xWplusb1pos = nn.MatrixVectorAdd(graph, xdotWpos, self.b1)
        # relu_xdotWpos = nn.ReLU(graph, xWplusb1pos)
        #
        # second_layerpos = nn.MatrixMultiply(graph, relu_xdotWpos, self.W2)
        # second_layerplusbpos = nn.MatrixVectorAdd(graph, second_layerpos, self.b2)
        #
        # xdotWneg = nn.MatrixMultiply(graph, input_negx, self.W1)
        # xWplusb1neg = nn.MatrixVectorAdd(graph, xdotWneg, self.b1)
        # relu_xdotWneg = nn.ReLU(graph, xWplusb1neg)
        # second_layerneg = nn.MatrixMultiply(graph, relu_xdotWneg, self.W2)
        # second_layerplusbneg = nn.MatrixVectorAdd(graph, second_layerneg, self.b2)
        # input_flipper = nn.Input(graph, -2*graph.get_output(second_layerplusbneg))
        # neg_second_layerplusbneg = nn.Add(graph, second_layerplusbneg,input_flipper)
        # make_odd = nn.Add(graph, second_layerplusbpos,neg_second_layerplusbneg)
        graph = nn.Graph([self.W1, self.W2, self.b1, self.b2])
        input_x = nn.Input(graph, x)
        input_negx = nn.Input(graph, np.dot(np.array(-1.0), x))

        xdotWpos = nn.MatrixMultiply(graph, input_x, self.W1)
        xWplusb1pos = nn.MatrixVectorAdd(graph, xdotWpos, self.b1)
        relu_xdotWpos = nn.ReLU(graph, xWplusb1pos)

        second_layerpos = nn.MatrixMultiply(graph, relu_xdotWpos, self.W2)
        second_layerplusbpos = nn.MatrixVectorAdd(graph, second_layerpos, self.b2)

        xdotWneg = nn.MatrixMultiply(graph, input_negx, self.W1)
        xWplusb1neg = nn.MatrixVectorAdd(graph, xdotWneg, self.b1)
        relu_xdotWneg = nn.ReLU(graph, xWplusb1neg)
        second_layerneg = nn.MatrixMultiply(graph, relu_xdotWneg, self.W2)
        second_layerplusbneg = nn.MatrixVectorAdd(graph, second_layerneg, self.b2)
        # input_flipper = nn.Input(graph, -2 * graph.get_output(second_layerplusbneg))
        input_flipper=nn.Input(graph,np.array(-1.0))
        neg_second_layerplusbneg = nn.MatrixMultiply(graph,input_flipper,second_layerplusbneg)
        make_odd = nn.Add(graph, second_layerplusbpos, neg_second_layerplusbneg)

        # graph = nn.Graph([self.W1pos, self.W2pos, self.b1pos, self.b2pos,self.W1neg, self.W2neg, self.b1neg, self.b2neg])
        # input_x = nn.Input(graph, x)
        # input_negx = nn.Input(graph, np.dot(np.array(-1.0), x))
        #
        # xdotWpos = nn.MatrixMultiply(graph, input_x, self.W1pos)
        # xWplusb1pos = nn.MatrixVectorAdd(graph, xdotWpos, self.b1pos)
        # relu_xdotWpos = nn.ReLU(graph, xWplusb1pos)
        #
        # second_layerpos = nn.MatrixMultiply(graph, relu_xdotWpos, self.W2pos)
        # second_layerplusbpos = nn.MatrixVectorAdd(graph, second_layerpos, self.b2pos)
        #
        # xdotWneg = nn.MatrixMultiply(graph, input_negx, self.W1neg)
        # xWplusb1neg = nn.MatrixVectorAdd(graph, xdotWneg, self.b1neg)
        # relu_xdotWneg = nn.ReLU(graph, xWplusb1neg)
        # second_layerneg = nn.MatrixMultiply(graph, relu_xdotWneg, self.W2neg)
        # second_layerplusbneg = nn.MatrixVectorAdd(graph, second_layerneg, self.b2neg)
        # input_flipper = nn.Input(graph, -2 * graph.get_output(second_layerplusbneg))
        # neg_second_layerplusbneg = nn.Add(graph, second_layerplusbneg, input_flipper)
        # make_odd = nn.Add(graph, second_layerplusbpos, neg_second_layerplusbneg)


        self.graphy = graph
        if y is not None :
            # At training time, the correct output `y` is known.
            # Here, you should construct a loss node, and return the nn.Graph
            # that the node belongs to. The loss node must be the last node
            # added to the graph.
            input_y = nn.Input(graph, y)
            loss = nn.SquareLoss(graph, make_odd, input_y)
            self.graphy=graph
            return self.graphy
        else:
            # At test time, the correct output is unknown.
            # You should instead return your model's prediction as a numpy array
            #print(self.graphy.get_output(self.graphy.get_nodes()[-2]))
            #print(graph.get_output(neg_second_layerplusbneg))
            #print(graph.get_output(second_layerplusbpos))
            #print(graph.get_output(xdotWpos))
            #print(graph.get_output(xdotWneg))
            return self.graphy.get_output(self.graphy.get_nodes()[-1])

class DigitClassificationModel(Model):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_digit_classification

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        self.learning_rate = .8
        self.hidden_size = 2100
        self.batch_size = 16
        self.W1, self.W2, self.b1, self.b2,self.W3,self.b3 = nn.Variable(784, self.hidden_size), nn.Variable(self.hidden_size,10), nn.Variable(self.hidden_size), nn.Variable(10),nn.Variable(self.hidden_size,self.hidden_size),nn.Variable(self.hidden_size)
    def run(self, x, y=None):
        """
        Runs the model for a batch of examples.

        The correct labels are known during training, but not at test time.
        When correct labels are available, `y` is a (batch_size x 10) numpy
        array. Each row in the array is a one-hot vector encoding the correct
        class.

        model should predict a (batch_size x 10) numpy array of scores,
        where higher scores correspond to greater probability of the image
        belonging to a particular class.  uses `nn.SoftmaxLoss` as
        training loss.

        Inputs:
            x: a (batch_size x 784) numpy array
            y: a (batch_size x 10) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 10) numpy array of scores (aka logits)
        """
        "*** YOUR CODE HERE ***"
        graph = nn.Graph([self.W1, self.W2, self.b1, self.b2,self.W3,self.b3])
        input_x = nn.Input(graph, x)
        xdotWpos = nn.MatrixMultiply(graph, input_x, self.W1)
        xWplusb1pos = nn.MatrixVectorAdd(graph, xdotWpos, self.b1)
        relu_xdotWpos = nn.ReLU(graph, xWplusb1pos)

        inner_layer = nn.MatrixMultiply(graph, relu_xdotWpos, self.W3)
        innerlayerplusb = nn.MatrixVectorAdd(graph, inner_layer, self.b3)
        innerrelu = nn.ReLU(graph, innerlayerplusb)

        second_layerpos = nn.MatrixMultiply(graph, innerrelu, self.W2)
        second_layerplusbpos = nn.MatrixVectorAdd(graph, second_layerpos, self.b2)





        if y is not None:
            "*** YOUR CODE HERE ***"
            input_y = nn.Input(graph, y)
            loss = nn.SquareLoss(graph, second_layerplusbpos, input_y)
            return graph
        else:
            return graph.get_output(second_layerplusbpos)


class DeepQModel(Model):
    """
    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.

    
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_rl
        self.learning_rate=.01
        self.num_actions = 2
        self.state_size = 4
        self.hidden_size = 30
        #self.batch_size = 16
        self.W1, self.W2, self.b1, self.b2 = nn.Variable(4, self.hidden_size), nn.Variable(self.hidden_size,2), nn.Variable(self.hidden_size), nn.Variable(2)
        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture


    def run(self, states, Q_target=None):
        """
        Runs the DQN for a batch of states.

        The DQN takes the state and computes Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]

        When Q_target == None, return the matrix of Q-values currently computed
        by the network for the input states.

        When Q_target is passed, it will contain the Q-values which the network
        should be producing for the current states. You must return a nn.Graph
        which computes the training loss between your current Q-value
        predictions and these target values, using nn.SquareLoss.

        Inputs:
            states: a (batch_size x 4) numpy array
            Q_target: a (batch_size x 2) numpy array, or None
        Output:
            (if Q_target is not None) A nn.Graph instance, where the last added
                node is the loss
            (if Q_target is None) A (batch_size x 2) numpy array of Q-value
                scores, for the two actions
        """
        
        graph = nn.Graph([self.W1, self.W2, self.b1, self.b2])
        input_x = nn.Input(graph, states)
        xdotWpos = nn.MatrixMultiply(graph, input_x, self.W1)
        xWplusb1pos = nn.MatrixVectorAdd(graph, xdotWpos, self.b1)
        relu_xdotWpos = nn.ReLU(graph, xWplusb1pos)


        second_layerpos = nn.MatrixMultiply(graph, relu_xdotWpos, self.W2)
        second_layerplusbpos = nn.MatrixVectorAdd(graph, second_layerpos, self.b2)


        if Q_target is not None:
            input_y = nn.Input(graph, Q_target)
            loss = nn.SquareLoss(graph, second_layerplusbpos, input_y)
            return graph
        else:
            
            return graph.get_output(second_layerplusbpos)

    def get_action(self, state, eps):
        """
        Select an action for a single state using epsilon-greedy.

        Inputs:
            state: a (1 x 4) numpy array
            eps: a float, epsilon to use in epsilon greedy
        Output:
            the index of the action to take (either 0 or 1, for 2 actions)
        """
        if np.random.rand() < eps:
            return np.random.choice(self.num_actions)
        else:
            scores = self.run(state)
            return int(np.argmax(scores))


class LanguageIDModel(Model):
    """
    A model for language identification at a single-word granularity.

    
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_lang_id

        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]
        self.hidden_size=2100
        self.learning_rate=.01
       

        self.W1, self.W2, self.b1, self.b2,self.W3 = nn.Variable(self.num_chars, self.hidden_size), nn.Variable(self.hidden_size,47), nn.Variable(self.hidden_size), nn.Variable(47),nn.Variable(47,5)
    def run(self, xs, y=None):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        (batch_size x self.num_chars) numpy array, where every row in the array
        is a one-hot vector encoding of a character. For example, if we have a
        batch of 8 three-letter words where the last word is "cat", we will have
        xs[1][7,0] == 1. Here the index 0 reflects the fact that the letter "a"
        is the inital (0th) letter of our combined alphabet for this task.

        The correct labels are known during training, but not at test time.
        When correct labels are available, `y` is a (batch_size x 5) numpy
        array. Each row in the array is a one-hot vector encoding the correct
        class.

        model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node that represents a (batch_size x hidden_size)
        array, for your choice of hidden_size. It should then calculate a
        (batch_size x 5) numpy array of scores, where higher scores correspond
        to greater probability of the word originating from a particular
        language. uses `nn.SoftmaxLoss` as your training loss.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a (batch_size x self.num_chars) numpy array
            y: a (batch_size x 5) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 5) numpy array of scores (aka logits)

        
        """
        batch_size = xs[0].shape[0]
        def nueral_net(graph,pass_in):




            xdotWpos = nn.MatrixMultiply(graph, pass_in, self.W1)
            xWplusb1pos = nn.MatrixVectorAdd(graph, xdotWpos, self.b1)
            relu_xdotWpos = nn.ReLU(graph, xWplusb1pos)

            second_layerpos = nn.MatrixMultiply(graph, relu_xdotWpos, self.W2)
            second_layerplusbpos = nn.MatrixVectorAdd(graph, second_layerpos, self.b2)


            return graph,second_layerplusbpos

        graph = nn.Graph([self.W1, self.W2, self.b1, self.b2,self.W3])
        h=nn.Input(graph,np.random.rand(batch_size,self.num_chars))
        #print(h.data.shape)

        for i in range(len(xs)):
            char=xs[i]
            input_char=nn.Input(graph,char)
            #print(input_char.data.shape)
            pass_in=nn.Add(graph,input_char,h)
            graph,h=nueral_net(graph,pass_in)

        h = nn.MatrixMultiply(graph, h, self.W3)
        

        if y is not None:

            y_input = nn.Input(graph, y)
            loss = nn.SoftmaxLoss(graph, h, y_input)
            return graph
        else:
            return graph.get_output(h)
