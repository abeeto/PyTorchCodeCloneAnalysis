from collections import deque
import random
import torch
import pickle
import heapq


class MaxHeapObj(object):
    def __init__(self, val):
        self.val = val

    def __lt__(self, other):
        return self.val > other.val

    def __eq__(self, other):
        return self.val == other.val

    def __str__(self):
        return str(self.val)


class MinHeap(object):
    def __init__(self):
        self.h = []

    def heappush(self, x):
        heapq.heappush(self.h, x)

    def heappop(self):
        return heapq.heappop(self.h)

    def __getitem__(self, i):
        return self.h[i]

    def __len__(self):
        return len(self.h)

    def remove(self, v):
        self.h.remove(v)

    def heaptop(self):
        if len(self.h) > 0:
            return self.h[0]
        else:
            return None


class MaxHeap(MinHeap):
    def heappush(self, x):
        heapq.heappush(self.h, MaxHeapObj(x))

    def heappop(self):
        return heapq.heappop(self.h).val

    def __getitem__(self, i):
        return self.h[i].val

    def remove(self, v):
        self.h.remove(MaxHeapObj(v))


class ReplayBuffer:
    bufferFname = 'replayBuffer.pth.tar'

    def __init__(self, dequeSize, batchSize):
        # init deque
        self.memory = deque(maxlen=dequeSize)
        # init heaps
        self.goodReward = MinHeap()
        self.badReward = MaxHeap()
        # init batch size
        self.batchSize = batchSize

    def getBufferFname(self):
        return self.bufferFname

    def getBufferSize(self):
        return len(self.memory)

    def loadFromFile(self, filepath):
        fileObj = open(filepath, 'rb')
        self.memory = pickle.load(fileObj)
        fileObj.close()

    # keeps on standard(constant) order: state, action, reward, next_state
    # no matter what is the actual order inside elem
    def extractBufferElement(self, elem):
        reward, state, action, next_state = elem
        return state, action, reward, next_state

    # builds element order, given the four: state, action, reward, next_state
    def buildBufferElement(self, state, action, reward, next_state):
        return reward, state, action, next_state

    def remember(self, save_path, state, action, reward, next_state):
        element = self.buildBufferElement(state, action, reward, next_state)

        if len(self.memory) == self.memory.maxlen:
            vOut = self.memory[0]
            # remove oldest deque element from heaps
            if vOut >= self.goodReward.heaptop():
                self.goodReward.remove(vOut)
            else:
                self.badReward.remove(vOut)

        # add element to the relevant heap
        if (len(self.goodReward) > 0) and (element >= self.goodReward.heaptop()):
            self.goodReward.heappush(element)
        else:
            self.badReward.heappush(element)

        # balance heaps
        if abs(len(self.goodReward) - len(self.badReward)) > 1:
            if len(self.goodReward) > len(self.badReward):
                vOut = self.goodReward.heappop()
                self.badReward.heappush(vOut)
            else:
                vOut = self.badReward.heappop()
                self.goodReward.heappush(vOut)

        # add element to deque
        self.memory.append(element)
        # save to file
        fileObj = open('{}/{}'.format(save_path, self.bufferFname), 'wb')
        pickle.dump(self.memory, fileObj)
        fileObj.close()

    def sample(self):
        # Sample trainSet from the memory
        trainSet = random.sample(self.goodReward.h, min(int(self.batchSize / 2), len(self.goodReward)))
        tempSet = random.sample(self.badReward.h, min(self.batchSize - len(trainSet), len(self.badReward)))
        trainSet.extend([t.val for t in tempSet])
        del tempSet

        # state, action & reward example
        state, action, _, _ = self.extractBufferElement(trainSet[0])

        trainState = torch.zeros(len(trainSet), state.size(0)).type_as(state)
        trainNextState = torch.zeros(trainState.size()).type_as(state)
        trainAction = torch.zeros(len(trainSet), action.size(0)).type_as(action)
        trainReward = torch.zeros(len(trainSet), 1).type_as(state)
        for i, element in enumerate(trainSet):
            state, action, reward, next_state = self.extractBufferElement(element)
            trainState[i] = state
            trainAction[i] = action
            trainReward[i] = reward
            trainNextState[i] = next_state

        return trainState, trainAction, trainReward, trainNextState
