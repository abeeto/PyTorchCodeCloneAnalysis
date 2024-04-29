# Some code is taken from PyTorch Website. The original code was from Robert Guthrie, further modifications are done by Sohail Ahmed Khan.

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

######################################################################

word_to_ix = {"hello": 0, "world": 1}
embeds = nn.Embedding(2, 5)  # 2 words in vocab, 5 dimensional embeddings
lookup_tensor = torch.LongTensor([word_to_ix["hello"]])
hello_embed = embeds(autograd.Variable(lookup_tensor))

######################################################################
# An Example: N-Gram Language Modeling
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Recall that in an n-gram language model, given a sequence of words
# :math:`w`, we want to compute
#
# .. math::  P(w_i | w_{i-1}, w_{i-2}, \dots, w_{i-n+1} )
#
# Where :math:`w_i` is the ith word of the sequence.
#
# In this example, we will compute the loss function on some training
# examples and update the parameters with backpropagation.
#

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
# We will use Shakespeare Sonnet 2
test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()
# we should tokenize the input, but we will ignore that for now
# build a list of tuples.  Each tuple is ([ word_i-2, word_i-1 ], target word)
# trigrams = [([test_sentence[i], test_sentence[i + 1]], test_sentence[i + 2])
#             for i in range(len(test_sentence) - 2)]
# print the first 3, just so you can see what they look like
# print(trigrams[:3])



sentences = ['<s> The mathematician ran . </s>', '<s> The mathematician ran to the store . </s>', 
'<s> The physicist ran to the store . </s>', '<s> The philosopher thought about it . </s>',
'<s> The mathematician solved the open problem . </s>']

vocab = []
sents = [line.split() for line in sentences]
trigrams = []
for j in range(len(sents)):
    vocab.extend(sents[j])
    trigrams.extend([([sents[j][i], sents[j][i + 1]], sents[j][i + 2]) for i in range(len(sents[j]) - 2)])
vocab = list(set(vocab))
word_to_ix = {word: i for i, word in enumerate(vocab)}


class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.05)

for epoch in range(35):
    total_loss = torch.Tensor([0])
    for context, target in trigrams:

        # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
        # into integer indices and wrap them in variables)
        context_idxs = [word_to_ix[w] for w in context]
        context_var = autograd.Variable(torch.LongTensor(context_idxs))
        # Step 2. Recall that torch *accumulates* gradients. Before passing in a
        # new instance, you need to zero out the gradients from the old
        # instance
        model.zero_grad()
        # Step 3. Run the forward pass, getting log probabilities over next
        # words
        log_probs = model(context_var)

        # Step 4. Compute your loss function. (Again, Torch wants the target
        # word wrapped in a variable)
        loss = loss_function(log_probs, autograd.Variable(
            torch.LongTensor([word_to_ix[target]])))
        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        optimizer.step()
        total_loss += loss.data
    losses.append(total_loss)
# print(losses)  # The loss decreased every iteration over the training data!
# print(log_probs)
# print(loss)

# TEST function for <s> The mathematician ran to the store . </s>
i = 0
for i in range(5):
	print('\n', '\t\tPass # ', i+1)
	trigrams = [([sents[1][i], sents[1][i + 1]], sents[1][i + 2]) for i in range(len(sents[1]) - 2)]
	for context, target in trigrams:
		context_idxs = [word_to_ix[w] for w in context]
		context_var = autograd.Variable(torch.LongTensor(context_idxs))
		log_probs = model(context_var)
		max_index = int(torch.argmax(log_probs))
		print('Expected Word: ' + target + '\tPredicted Word: ' + vocab[max_index])


s = ['<s> The physicist solved the open problem . </s>', '<s> The philosopher solved the open problem . </s>']
trigrams_physicist = []
trigrams_philospher = []
sents_phys = [s[0].split()]
sents_philospher = [s[1].split()]

#Trigrams for Test Sentences
for j in range(len(sents_phys)):
    trigrams_physicist.extend([([sents_phys[j][i], sents_phys[j][i + 1]], sents_phys[j][i + 2]) for i in range(len(sents_phys[j]) - 2)])
    trigrams_philospher.extend([([sents_philospher[j][i], sents_philospher[j][i + 1]], sents_philospher[j][i + 2]) for i in range(len(sents_philospher[j]) - 2)])


# Calculating loss for test sentence having 'physicist'
phys_total_loss = torch.Tensor([0])
phys_total_prob = torch.Tensor([1])
for context, target in trigrams_physicist:
	context_idxs = [word_to_ix[w] for w in context]
	context_var = autograd.Variable(torch.LongTensor(context_idxs))
	log_probs = model(context_var)
	phys_total_prob *= torch.max(log_probs)
	phys_total_loss += loss.data
	# max_index = int(torch.argmax(log_probs))


# Calculating loss for test sentence having 'philosopher'
philosopher_total_loss = torch.Tensor([0])
philosopher_total_prob = torch.Tensor([1])
for context, target in trigrams_philospher:
	context_idxs = [word_to_ix[w] for w in context]
	context_var = autograd.Variable(torch.LongTensor(context_idxs))
	log_probs = model(context_var)
	loss = loss_function(log_probs, autograd.Variable(
            torch.LongTensor([word_to_ix[target]])))
	philosopher_total_prob *= torch.max(log_probs)
	philosopher_total_loss += loss.data
	# max_index = int(torch.argmax(log_probs))


print('\n ---------------- Results according to Model\'s Predictions using Probabilities ----------------\n')
print('Probability for Physicist: ' , phys_total_prob , '\t Probability for Philosopher: ' , philosopher_total_prob)
if phys_total_prob >= philosopher_total_prob:
	print('Predicted Sentence: ->\t' + s[0])
else:
	print('Predicted Sentence: ->\t' + s[1])


print('\n\n\n\n ---------------- Results according to Model\'s Predictions using Loss ----------------\n')
print('Loss for Physicist: ' , phys_total_loss , '\t Loss for Philosopher: ' , philosopher_total_loss)
if phys_total_loss <= philosopher_total_loss:
	print('Predicted Sentence: ->\t' + s[0])

else:
	print('Predicted Sentence: ->\t' + s[1])


phys_lookup_tensor = torch.LongTensor([word_to_ix["physicist"]])
philosopher_lookup_tensor = torch.LongTensor([word_to_ix["philosopher"]])
mathematician_lookup_tensor = torch.LongTensor([word_to_ix["mathematician"]])

phy_embed = model.embeddings(autograd.Variable(phys_lookup_tensor)).view((1, -1))
phil_embed = model.embeddings(autograd.Variable(philosopher_lookup_tensor)).view((1, -1))
mathematician_embed = model.embeddings(autograd.Variable(mathematician_lookup_tensor)).view((1, -1))

cos_phy = nn.CosineSimilarity(dim=-1)
phy = (cos_phy(mathematician_embed, phy_embed))

cos_phil = nn.CosineSimilarity(dim=-1)
phil = (cos_phil(mathematician_embed, phil_embed))


print('\n\n\n\n ---------------- Results according to CosineSimilarity ----------------\n')

if phy >= phil:
	print('------> mathematician and physicist are more similar according to CosineSimilarity') 
	print('Predicted Sentence: ->\t' + s[0])
else:
	print('\n------> mathematician and philosopher are more similar according to CosineSimilarity') 
	print('Predicted Sentence: ->\t' + s[1])

