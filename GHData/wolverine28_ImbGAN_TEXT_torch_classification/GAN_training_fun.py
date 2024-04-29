import torch
import torch.nn as nn

from rollout import Rollout
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

from AGNews_GAN_main import GANLoss, eval_epoch, generate_samples, discriminator_train_epoch

GENERATED_NUM = 10000
def train_GAN_SUB(g_hidden_dim, generator, netM, netM_optimizer, discriminator, dis_optimizer,train_loader, SOS, nEPOCH, BATCH_SIZE, SEQ_LEN, vocab, use_cuda=True):
    
    # Adversarial Training
    criterion = nn.NLLLoss(reduction='sum')
    if use_cuda:
        criterion = criterion.cuda()

    for total_batch in range(nEPOCH):
        ones = Variable(torch.ones(BATCH_SIZE).long().cuda())

        # update discriminator
        for _ in range(2):
            z = Variable(torch.randn((BATCH_SIZE, g_hidden_dim*2))).cuda()
            sub_z = netM(z)
            fake_samples = generate_samples(generator, sub_z, BATCH_SIZE, GENERATED_NUM/2, SOS)
            
            fake_samples = torch.Tensor(fake_samples).long()
            fake_dataset = TensorDataset(fake_samples)
            fake_loader = DataLoader(fake_dataset,batch_size=BATCH_SIZE)

            for _ in range(1):
                loss = discriminator_train_epoch(discriminator, train_loader, fake_loader, criterion, dis_optimizer)
                print('Epoch [%d], Discriminator loss: %f' % (total_batch, loss))

        # update generator
        for i, batch in enumerate(train_loader, 0):
            target = batch[1]
            data = batch[0]
            if use_cuda:
                data, target = data.cuda(), target.float().cuda()

            netM_optimizer.zero_grad()
            # z = Variable(torch.randn((1, BATCH_SIZE, g_hidden_dim)))
            z = Variable(torch.randn((BATCH_SIZE, g_hidden_dim*2))).cuda()
            sub_z = netM(z)
            fake_samples = torch.tensor(generate_samples(generator, sub_z, BATCH_SIZE, BATCH_SIZE, SOS)).long().cuda()
            gen_loss = criterion(discriminator(fake_samples), ones)
            gen_loss.backward()
            netM_optimizer.step()


