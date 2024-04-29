import torch

"""
For each row it samples uniformly between a
minimal and maximal element; removes collisions
""" 
class ShardedNegUniformSampler:
    def __init__(self,num_sampled):
        self.num_sampled = num_sampled

    """
    (true positive items, minimal id for sampling, maximal id
    for sampling)

    =>

    negative Samples with shape (inputIds.size, num_sampled)
    """
    def getSample(self, inputIds, minIds, maxIds):
        # create the rescaling width; need to add 1.0 as sampler
        # in [0,1)
        width = (maxIds-minIds+1.0).type(torch.FloatTensor)
        sampled0 = torch.rand(size=(minIds.size()[0], self.num_sampled))
        # rescale the sampled0 by width
        sampled1 = width.view(-1,1)*sampled0
        # add the minimum element so we fall in the sampled items
        sampled2 = minIds.view(-1,1).type(torch.FloatTensor)+sampled1
        # floor to integer
        sampled3 = sampled2.floor().type(torch.LongTensor)
        # increase when there is a collision
        sampled4 = torch.where(sampled3 == inputIds.view(-1,1),sampled3+1,sampled3)
        # decrease again when we go above the maximum allowed value
        sampled5 = torch.where(sampled4 > maxIds.view(-1,1),
                               minIds.view(-1,1).expand(-1,self.num_sampled),
                               sampled4)

        return sampled5
    






