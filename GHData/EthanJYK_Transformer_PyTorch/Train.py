#%% Setup Parameters and Models
# Training -------------------------------------------------------------------#
# parameters
torch.manual_seed(999999)
torch.cuda.manual_seed_all(999999)
d_model = 512
heads = 8
N = 6
mask = True
dropout = 0.1
padding_idx = 1 # FR_TEXT.vocab.stoi['<pad>'], EN_TEXT.vocab.stoi['<pad>'] == 1
epsilon = 0.1 # Label Smoothing epsilon
max_seq_len = 200
src_vocab = len(EN_TEXT.vocab)
trg_vocab = len(FR_TEXT.vocab)

# create a model
model = Transformer(src_vocab, trg_vocab, d_model, N, heads, padding_idx, max_seq_len, dropout)
model.cuda()

# Xavier initialization
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
# except Embedding layers - random normal(0, 1)
nn.init.normal_(model.embed1.embed.weight, mean=0, std=1)
nn.init.normal_(model.embed2.embed.weight, mean=0, std=1)

# optimizer
opt_adam = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.997), eps=1e-9)
opt_warmup = LRScheduler(d_model, 0.5, 15, step_from=0)
#-----------------------------------------------------------------------------#



# Train with validation function (all epochs) --------------------------------#
def train_with_val(epochs, print_every=100, optimizer=opt_adam):
    start = time.time()
    temp = start

    total_loss = 0
    total_val_loss = 0

    optimizer = optimizer

    iterator = {'train':train_iter, 'val':val_iter}

    train_iter_length = len(list(enumerate(train_iter)))
    val_iter_length = len(list(enumerate(val_iter)))

    for epoch in range(epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True) # set model to training mode
                for param in model.parameters(): param.requires_grad = True # turn on gradients
                print('[Train]')
            else:
                model.train(False) # set model to evaluating mode
                for param in model.parameters(): param.requires_grad = False # turn off gradients

            # load data
            for i, batch in enumerate(iterator[phase]):
                src = batch.English.transpose(0, 1).cuda()
                trg = batch.French.transpose(0, 1).cuda()
                trg_input = trg[:, :-1]
                targets = trg[:, 1:].contiguous().view(-1) # remove <sos> and

                # run and get loss
                preds = model(src, trg_input, mask=True)
                loss = LSKLDivLoss(preds.view(-1, preds.size(-1)), targets, trg_vocab, 0.1 ,padding_idx)

                # backpropagate and update
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss
                    n = 0

                    # print out min-training results
                    if (i + 1) % print_every == 0:
                        n = print_every
                    if (i + 1) == train_iter_length:
                        n = train_iter_length % print_every
                    if n > 0:
                        loss_avg = total_loss / n
                        print("Elapsed = %dm, epoch %d, iter = %d, loss = %.8f, %ds per %d iters"
                              % ((time.time() - start) // 60, epoch + 1, i + 1, loss_avg, time.time() - temp, n))
                        total_loss = 0
                        temp = time.time()
                        n = 0

                # record validation loss (ouput per epoch)
                else:
                    total_val_loss += loss
                    # print out min-training results

            # print out validation results
            if phase == 'val':
                print('[Val]')
                val_loss_avg = total_val_loss / val_iter_length
                print("Elapsed = %dm, epoch %d, val_loss = %.8f, %ds per epoch"
                      % ((time.time() - start) // 60, epoch + 1, val_loss_avg, time.time() - temp))
                total_val_loss = 0
                temp = time.time()
#-----------------------------------------------------------------------------#



# see Test set loss ----------------------------------------------------------#
def test():

    model.eval()
    start = time.time()
    temp = start
    total_loss = 0
    test_iter_length = len(list(enumerate(test_iter)))

    with torch.no_grad():
        for i, batch in enumerate(test_iter):
            src = batch.English.transpose(0, 1).cuda()
            trg = batch.French.transpose(0, 1).cuda()

            trg_input = trg[:, :-1]
            targets = trg[:, 1:].contiguous().view(-1)

            preds = model(src, trg_input, mask=True)

            loss = LSKLDivLoss(preds.view(-1, preds.size(-1)), targets, trg_vocab, 0.1, padding_idx)

            total_loss += loss

        # print out validation results
        print('[Test]')
        loss_avg = total_loss / test_iter_length
        print("Elapsed = %dm, test_loss = %.8f, %ds per epoch"
              % ((time.time() - start) // 60, loss_avg, time.time() - temp))
#-----------------------------------------------------------------------------#



# Translation Test -----------------------------------------------------------#
def translate(model, src, max_len = 200, mask=True, custom_string=False):

    model.eval()

    if custom_string == True:
        src = tokenize_en(src)
        src = Variable(torch.LongTensor([[EN_TEXT.vocab.stoi[tok] for tok in src]])).cuda()

    src = model.embed1(src)
    src_mask = get_mask(src, mask)
    src = model.pe1(src)
    memory = model.encoder(src, src_mask)

    outputs = torch.zeros(max_len, dtype=torch.long).cuda()
    outputs[0] = torch.LongTensor([FR_TEXT.vocab.stoi['<sos>']])

    for i in range(1, max_len):

        trg = model.embed2(outputs[:i].unsqueeze(0))
        trg_mask = get_mask(trg, mask)
        trg = model.pe2(trg)
        out = model.out(model.decoder(trg, memory, src_mask, trg_mask))
        out = F.softmax(out, dim=-1)
        val, ix = out[:, -1].data.topk(1)

        outputs[i] = ix[0][0]
        if ix[0][0] == FR_TEXT.vocab.stoi['<eos>']:
            break

    return ' '.join([FR_TEXT.vocab.itos[ix] for ix in outputs[:i]])
#-----------------------------------------------------------------------------#


#%% train
train_with_val(epochs=1, optimizer=opt_adam)


#%% test
test()


#%% translation test
for i in range(10):
    print("[%d]" % (i + 1))
    print(en_sample[i])
    print(translate(model=model, src=en_sample[i], custom_string=True))


#%% translation test with a custom string
print(translate(model, "We should make something like that.", custom_string=True))


#%% save and load model
torch.save(model.state_dict(), 'params.300e.pth')
model.load_state_dict(torch.load('params.300e.pth'))
#-----------------------------------------------------------------------------#
