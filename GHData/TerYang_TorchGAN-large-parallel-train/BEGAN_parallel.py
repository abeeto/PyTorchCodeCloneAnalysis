import utils, torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from dataloader import dataloader
from nets import *
from tensorboardX import SummaryWriter
import json
from utils import *
# class generator(nn.Module):
#     # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
#     # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S

#             nn.BatchNorm1d(1024),
#             nn.ReLU(),
#             nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
#             nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
#             nn.ReLU(),
#         )
#         self.deconv = nn.Sequential(
#             nn.ConvTranspose2d(128, 64, 4, 2, 1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
#             nn.Tanh(),
#         )
#         utils.initialize_weights(self)
#
#     def forward(self, input):
#         x = self.fc(input)#     def __init__(self, input_dim=100, output_dim=1, input_size=32):
# #         super(generator, self).__init__()
# #         self.input_dim = input_dim
# #         self.output_dim = output_dim
# #         self.input_size = input_size
# #
# #         self.fc = nn.Sequential(
# #             nn.Linear(self.input_dim, 1024),
#         x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
#         x = self.deconv(x)
#
#         return x
#
# class discriminator(nn.Module):
#     # It must be Auto-Encoder style architecture
#     # Architecture : (64)4c2s-FC32-FC64*14*14_BR-(1)4dc2s_S
#     def __init__(self, input_dim=1, output_dim=1, input_size=32):
#         super(discriminator, self).__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.input_size = input_size
#
#         self.conv = nn.Sequential(
#             nn.Conv2d(self.input_dim, 64, 4, 2, 1),
#             nn.ReLU(),
#         )
#         self.fc = nn.Sequential(
#             nn.Linear(64 * (self.input_size // 2) * (self.input_size // 2), 32),
#             nn.Linear(32, 64 * (self.input_size // 2) * (self.input_size // 2)),
#         )
#         self.deconv = nn.Sequential(
#             nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
#             #nn.Sigmoid(),
#         )
#
#         utils.initialize_weights(self)
#
#     def forward(self, input):
#         x = self.conv(input)
#         x = x.view(x.size()[0], -1)
#         x = self.fc(x)
#         x = x.view(-1, 64, (self.input_size // 2), (self.input_size // 2))
#         x = self.deconv(x)
#
#         return x

class BEGAN(object):
    # def __init__(self, args,data_loader,valdata):
    # def __init__(self, args):
    def __init__(self, data_loader,valdata,dataset_type,train_type):
        # parameters
        args = parse_args()
        self.dataset = dataset_type
        self.data_loader = data_loader
        self.valdata = valdata
        self.model_name = self.__class__.__name__ + '_' + train_type

        # parameters
        self.epoch = args.epoch
        self.sample_num = 100
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        # self.dataset = args.dataset
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        # self.model_name = args.gan_type
        self.input_size = args.input_size
        self.z_dim = 62
        self.gamma = 1
        self.lambda_ = 0.001
        self.k = 0.0
        self.lr_lower_boundary = 0.00002
        self.train_hist = {}
        self.M = {}

        # load dataset
        # self.data_loader = dataloader(self.dataset, self.input_size, self.batch_size)
        # data = self.data_loader.__iter__().__next__()[0]

        # self.data_loader = testToGAN(self.dataset,'train')

        # parallel run,train data and validate data
        # self.data_loader = data_loader
        # self.valdata = valdata
        # self.data_loader = DataloadtoGAN(self.dataset,'train')

        # add label
        # self.data_loader = DataloadtoGAN(self.dataset,mark='train', single_dataset=True)
        #
        # self.valdata = DataloadtoGAN(self.dataset,'validate')

        # 重置dataset
        # self.dataset = 'StepLR'
        data = next(iter(self.data_loader))[0]
        # self.model_name = self.model_name + '_sig_daset'

        # networks init
        self.G = generator(input_dim=self.z_dim, output_dim=data.shape[1], input_size=self.input_size)
        self.D = discriminator(input_dim=data.shape[1], output_dim=1, input_size=self.input_size)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=0.0002, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=0.0002, betas=(args.beta1, args.beta2))

        # Step LR
        # self.G_scheduler = optim.lr_scheduler.StepLR(self.G_optimizer, 20, gamma=0.1, last_epoch=-1)
        # self.D_scheduler = optim.lr_scheduler.StepLR(self.D_optimizer, 20, gamma=0.1, last_epoch=-1)

        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()

        print('---------- Networks architecture -------------')
        utils.print_network(self.G)
        utils.print_network(self.D)
        print('-----------------------------------------------')

        # fixed noise
        self.sample_z_ = torch.rand((self.batch_size, self.z_dim))
        if self.gpu_mode:
            self.sample_z_ = self.sample_z_.cuda()

        # label for cal loss
        # self.y_real_, self.y_fake_ = torch.ones(self.batch_size, 1), torch.zeros(self.batch_size, 1)
        self.y_real_, self.y_fake_ = torch.zeros(self.batch_size, 1), torch.ones(self.batch_size, 1)
        if self.gpu_mode:
            self.y_real_, self.y_fake_ = self.y_real_.cuda(), self.y_fake_.cuda()

        self.writer = SummaryWriter()#log_dir=log_dir,
        self.X = 0
        print('Training {},started at {}'.format(self.model_name, time.asctime(time.localtime(time.time()))),end=',')

    def train(self):
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []
        self.M['pre'] = []
        self.M['pre'].append(1)
        self.M['cur'] = []

        self.D.train()
        print('{} training start!!,data set:{},epoch:{}'.format(self.model_name, self.dataset,self.epoch))
        start_time = time.time()
        # url = os.path.join(self.save_dir, self.dataset, self.model_name)

        for epoch in range(self.epoch):
            # if epoch == 110:
            #     self.G = torch.load(os.path.join(url,'BEGAN_110_G.pkl'))
            #     self.D = torch.load(os.path.join(url,'BEGAN_110_D.pkl'))
            #     print('reload success!','*'*40)
            self.G.train()
            self.D.train()
            epoch_start_time = time.time()
            # for iter, (x_, _) in enumerate(self.data_loader):
            for iter, x_, in enumerate(self.data_loader):
                x_ = x_[0]

                if iter == self.data_loader.dataset.__len__() // self.batch_size:
                    break

                z_ = torch.rand((self.batch_size, self.z_dim))

                if self.gpu_mode:
                    x_, z_ = x_.cuda(), z_.cuda()

                # update D network
                self.D_optimizer.zero_grad()

                D_real = self.D(x_)
                D_real_loss = torch.mean(torch.abs(D_real - x_))

                G_ = self.G(z_)
                D_fake = self.D(G_)
                D_fake_loss = torch.mean(torch.abs(D_fake - G_))

                D_loss = D_real_loss - self.k * D_fake_loss
                self.train_hist['D_loss'].append(D_loss.item())

                D_loss.backward()
                self.D_optimizer.step()

                # update G network
                self.G_optimizer.zero_grad()

                G_ = self.G(z_)
                D_fake = self.D(G_)
                D_fake_loss = torch.mean(torch.abs(D_fake - G_))

                G_loss = D_fake_loss
                self.train_hist['G_loss'].append(G_loss.item())

                G_loss.backward()
                self.G_optimizer.step()

                # convergence metric
                temp_M = D_real_loss + torch.abs(self.gamma * D_real_loss - G_loss)

                # operation for updating k
                temp_k = self.k + self.lambda_ * (self.gamma * D_real_loss - G_loss)
                temp_k = temp_k.item()

                self.k = min(max(temp_k, 0), 1)
                self.M['cur'] = temp_M.item()

                if ((iter + 1) % 100) == 0:
                    self.writelog("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f, M: %.8f, k: %.8f,G_lr: %.8f,D_lr: %.8f"%
                          ((epoch + 1), (iter + 1), self.data_loader.dataset.__len__() // self.batch_size, D_loss.item(),
                           G_loss.item(), self.M['cur'], self.k, self.G_optimizer.param_groups[0]['lr'],self.D_optimizer.param_groups[0]['lr']))

                    # print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f, M: %.8f, k: %.8f" %
                    #       ((epoch + 1), (iter + 1), self.data_loader.dataset.__len__() // self.batch_size, D_loss.item(), G_loss.item(), self.M['cur'], self.k))

                    self.writer.add_scalar('G_loss', G_loss.item(), self.X)
                    # writer.add_scalar('G_loss' , -G_loss_D, X)
                    self.writer.add_scalar('D_loss', D_loss.item(), self.X)
                    self.writer.add_scalars('cross loss', {'G_loss': D_loss.item(),
                                                      'D_loss': D_loss.item()}, self.X)
                    self.X += 1

            if np.mean(self.M['pre']) < np.mean(self.M['cur']):
                pre_lr = self.G_optimizer.param_groups[0]['lr']
                self.G_optimizer.param_groups[0]['lr'] = max(self.G_optimizer.param_groups[0]['lr'] / 2.0,
                                                             self.lr_lower_boundary)
                self.D_optimizer.param_groups[0]['lr'] = max(self.D_optimizer.param_groups[0]['lr'] / 2.0,
                                                             self.lr_lower_boundary)
                print('M_pre: ' + str(np.mean(self.M['pre'])) + ', M_cur: ' + str(
                    np.mean(self.M['cur'])) + ', lr: ' + str(pre_lr) + ' --> ' + str(
                    self.G_optimizer.param_groups[0]['lr']))
            else:
                print('M_pre: ' + str(np.mean(self.M['pre'])) + ', M_cur: ' + str(np.mean(self.M['cur'])))
                self.M['pre'] = self.M['cur']

                self.M['cur'] = []

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            if epoch % 5 == 0:
                self.load_interval(epoch)

            # validate and schedule lr
            # acc_D = validate(self.D, None, self.valdata, None)#def validate(model,data_loader=None,data=None,label=None)
            acc_D = validate(self.D, self.valdata)#def validate(model,data_loader=None,data=None,label=None)

            # ReduceLROnPlateau
            # self.D_scheduler.step(acc_D)

            # reduce by step epoch
            # self.D_scheduler.step(epoch)
            self.D.cuda()
            # self.D.train()

            # schedule G lr
            # acc_G = self.validate_G(self.valdata.data.numpy().shape[0])
            acc_G = self.validate_G(self.valdata.dataset.tensors[0].shape[0]//2)
            # acc_G = self.validate_G(self.valdata.dataset.__len__//2)

            # ReduceLROnPlateau
            # self.G_scheduler.step(acc_D)
            # reduce by step epoch
            # self.G_scheduler.step(epoch)
            # self.G.train()

            # with torch.no_grad():
            #     self.visualize_results((epoch+1))

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
              self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        with open(os.path.join(save_dir, self.model_name + '_train_hist.json'), "a") as f:
            json.dump(self.train_hist, f)

        self.writer.export_scalars_to_json(os.path.join(save_dir, self.model_name + '.json'))
        self.writer.close()

        self.load_interval(epoch)

        # self.save()
        # utils.generate_animation(self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name,
        #                          self.epoch)

        utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)

    def load(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))

    def load_interval(self,epoch):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # 保存模型
        torch.save(self.G, os.path.join(save_dir, self.model_name + '_{}_G.pkl'.format(epoch)))#dictionary ['bias', 'weight']
        torch.save(self.D, os.path.join(save_dir, self.model_name + '_{}_D.pkl'.format(epoch)))

    def writelog(self, content):
        save_dir = os.path.join(os.getcwd(), self.save_dir, self.dataset, self.model_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_log = os.path.join(save_dir,'train_records.txt')
        with open(save_log,'a',encoding='utf-8') as f:
            f.writelines('\n'+content + '\n')
            print(content)

    def validate_G(self, size):
        # validate G
        self.G.eval()
        acc_G = 0
        sum_all = 0
        for i in range(size // 64):
            z_ = torch.rand((self.batch_size, self.z_dim))
            if self.gpu_mode:
                # x_, z_ = x_.cuda(), z_.cuda()
                z_ = z_.cuda()
            G_ = self.G(z_)
            # print('G_.shape:', G_.shape)
            D_fake = self.D(G_)
            # print(D_fake.__class__)
            D_fake = np.squeeze(D_fake.data.cpu().numpy(), axis=1)
            # D_fake = D_fake.tolist()
            f = lambda x: 1 if x > 0.5 else 0
            ll = list(map(f, D_fake.tolist()))
            acc_G += ll.count(1)
            sum_all += len(ll)
        zeros = sum_all - acc_G
        ones = acc_G
        print('G, size:%d,zeros:%d,ones:%d' % (sum_all, zeros, ones), end=',')
        print('acc:%.6f,judged as 1.' % (ones / sum_all))
        return ones / sum_all