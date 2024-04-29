##################################################
# Dynamic Programming Algorithms:
# * Forward Algorithm (for computing normalizer)
# * Sequence-Scoring Algorithm
# * Viterbi Decoding Algorithm
##################################################

import torch
import torch.nn as nn
from torch.autograd import Variable
from fns import argmax, log_sum_exp

__CUDA__ = torch.cuda.is_available()

def forward_alg(features, transitions, batch_size, num_labels, start_token, stop_token, use_cuda=__CUDA__):
    """
    Use the forward-algorithm to compute the normalizing value for a set of convolutional features.
    
    In the definition of the conditional probability induced by a CRF model, recall that

      p(labels|sequence) = p(Y|X) = [Z(X)]^{-1} * exp(score(y,x)).

    Here we compute the Z(X) function given a set of features Conv(X), i.e. assume that Z(X) = g(Conv(X));
    we use the expression given by the forward algorithm:

      Z(X) := Sum_over_Ys[alpha(Y,T)] where alpha(Y,t) := Prob of seeing a sequence with t-th element = Y

    N.B.: for computational efficiency, everything is done in log space; e.g. what would normally be
    division/multiplication is thus subtraction/addition due to the endomorphism induced by log().

    Args:
    * features: FloatTensor variable of size (seq, batch, num_labels).
    * transitions: an nn.Parameter object of size (num_labels, num_labels); transitions[i,j] gives the
      score associated to a transition *from* label j *to* label i.
    * batch_size: 
    * num_labels:
    * start_token:
    * stop_token:

    Returns:
    * final_alpha: FloatTensor variable of shape (batch); this is the normalizing value for each feature
    sequence.
    """
    ### initialize the alpha variable with all prob. mass placed on START token:
    init_alphas = torch.Tensor(batch_size, num_labels).fill_(-10000.)
    init_alphas[:, start_token] = 0.
    forward_var = Variable(init_alphas) # ~ (batch_size, num_labels)
    if use_cuda: forward_var = forward_var.cuda()

    ### iterate through the sequence of features and update the forward variable over time:
    # note that feat ~ (batch_size, num_labels):
    for feat in features:
        #-- alpha(-,t) for timestep t:
        alphas_t = []
            
        #-- compute alpha(y,t) for all possible next labels y:
        for label in range(num_labels):
            #--- emission score at time t ~ (batch, num_labels)
            emit_score = torch.stack([feat[:, label]] * num_labels, 1)
            #--- transiton score at time t ~ (batch, num_labels):
            trans_score = torch.stack([transitions[label]] * batch_size, 0)
            #--- next_label_var is alpha(-,t) in log-space:
            next_label_var = forward_var + trans_score + emit_score
            #--- update alpha with logsumexp(next_label_var) ~ (batch):
            alphas_t.append(log_sum_exp(next_label_var, use_cuda=use_cuda))
        #-- update forward variable:
        forward_var = torch.stack(alphas_t,1)

    ### compute final timestep by adding <STOP> score:
    forward_var += torch.stack([transitions[stop_token]] * batch_size, 0)
        
    ### perform final LogSumExp and return:
    return log_sum_exp(forward_var, use_cuda=use_cuda)


def score_sequences(features, labels, batch_size, transitions, start_token, stop_token, use_cuda=__CUDA__):
    """
    Given a sequence of convolution features and a proposed labelling of the features,
    score the labelling.
    
    Args:
    * features: the output of the featurizing convolutional network. A FloatTensor variable
    of shape (seq, batch, num_labels).
    * labels: the proposed labels for each timestep. LongTensor variable of shape (seq, batch).
    
    Returns:
    * score: FloatTensor variable of shape (batch); provides the score of each (features,labels)
    pair.
    """
    # score ~ (batch)
    score = Variable(torch.Tensor([0.] * batch_size))
    if use_cuda: score = score.cuda()

    # append <START> token to labels:
    _start_tokens = Variable(torch.LongTensor([start_token] * batch_size)).unsqueeze(0)
    if use_cuda: _start_tokens = _start_tokens.cuda()
    labels = torch.cat((_start_tokens, labels), 0)

    # loop over feature sequence and accumulate score:
    # (recall Score(x,y) == conv_features + transition_scores)
    for i, feat in enumerate(features):
        # add transition score:
        _transition_scores = [ transitions[labels.data[i+1,b], labels.data[i,b]] for b in range(batch_size) ]
        score += torch.stack(_transition_scores,0)
        # add emission score:
        _emission_scores = [ feat[b, labels.data[i+1,b]] for b in range(batch_size) ]
        score += torch.stack(_emission_scores,0)

    # add final <STOP> score to sequence score:
    _stop_transition_scores = [ transitions[stop_token, labels.data[-1,b]] for b in range(batch_size) ]
    score += torch.stack(_stop_transition_scores, 0)

    return score


def viterbi_decode(features, transitions, batch_size, num_labels, start_token, stop_token, use_cuda=__CUDA__):
    """
    Compute most likely sequence generated by features, using the Viterbi algorithm.

    N.B.: everything here is done in log-space, hence additions instead of multiplications
    thanks to the log-exp homomorphisms from (R,*) <-> (R,+).

    Args:
    * features: FloatTensor variable of shape (seq, batch, num_labels). The features
      coming from the ConvNet.
    * transitions:
    * batch_size:
    * num_labels:
    * start_token:
    * stop_token:

    Returns: a tuple (path_score, best_path) where:
    * path_score: score of the most probable path through features, of shape (batch).
    * best_path: the most probable labelling of the features; this is a list of batches,
      i.e. `[(batch_size)] * seq_length`, with values in `[0,num_labels)`.
    """
    ### initialize starting values:
    # a list of current backpointers:
    backpointers = []
    # initialize the forward variable in log-space with all weight on the start token:
    viterbi_var_0 = torch.Tensor(batch_size, num_labels).fill_(-10000.)
    viterbi_var_0[:, start_token] = 0.
    forward_var = Variable(viterbi_var_0)
    if use_cuda: forward_var = forward_var.cuda()

    ### loop over each timestep of the feature sequence and perform the following:
    # 1. compute most likely next label (for each seq in the batch);
    # 2. compute the score for that label;
    # 3. create/store backpointer at that timestep.
    for feat in features:
        #-- hold backpointers and viterbi variables at timestep t:
        bptrs_t = []
        viterbi_var_t = []

        #-- for each label, compute the score for that label at this timestep:
        for next_label in range(num_labels):
            # next_label_var[i] ~ viterbi var for label i @ prev step, plus transition score;
            # don't add emission score since that can be done later without looping
            _next_label_trans_scores = transitions[next_label, :].unsqueeze(0).expand_as(forward_var)
            next_label_var = forward_var + _next_label_trans_scores # ~ (batch, num_labels)
            best_label_ids = argmax(next_label_var) # ~ (batch)
            # update backpointers and viterbi variables:
            bptrs_t.append(best_label_ids) # best_label_ids ~ [(batch)]
            _next_tag_var_id = torch.stack([next_label_var[b,best_label_ids[b]] for b in range(batch_size)])
            viterbi_var_t.append(_next_tag_var_id)

        #-- add emission scores here instead of in the for loop (more numerically stable this way);
        #   also update forward_var.
        forward_var = torch.cat(viterbi_var_t,1) + feat
            
        #-- update backpointers with computed bptrs_t (so that we can backtrace at this step later):
        backpointers.append(bptrs_t)

    ### add a final transition to the <STOP> label:
    _stop_token_trans_scores = transitions[stop_token, :].unsqueeze(0).expand_as(forward_var)
    terminal_var = forward_var + _stop_token_trans_scores
    best_label_ids = argmax(terminal_var) # ~ (batch)

    ### score of the best path: ~ (batch)
    path_scores = torch.stack([terminal_var[b, best_label_ids[b]] for b in range(batch_size)], 0).squeeze(1)

    ### backtrack through the backpointers to compute the best paths for each batch:
    best_paths = viterbi_backtrack(batch_size, best_label_ids, backpointers, start_token)

    return path_scores, best_paths


def viterbi_backtrack(batch_size, best_label_ids, backpointers, start_token, use_cuda=__CUDA__):
    """
    Given a set of backpointers at each timestep (in reversed order), walk backwards through
    all backpointer IDs to compute the optimal decoded viterbi path.
    """
    ### backtrack through computed probas to get the most likely labelling:
    # initialize best path at most probable final ids:
    best_paths = [best_label_ids]
    # walk backwards through backpointers at each timestep:
    for bptrs_t in reversed(backpointers):
        best_label_ids = [bptrs_t[best_label_ids[b]][b] for b in range(batch_size)]
        best_paths.append(best_label_ids)

    ### remove <START> tag:
    start = best_paths.pop()
    assert start[0] == start_token # sanity check
        
    ### return best path and its score:
    best_paths.reverse()
    if use_cuda:
        return torch.cuda.LongTensor(best_paths)
    else:
        return torch.LongTensor(best_paths)
