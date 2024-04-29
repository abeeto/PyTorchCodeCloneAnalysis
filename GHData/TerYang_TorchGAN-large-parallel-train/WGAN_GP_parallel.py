import utils, torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
# from dataloader import dataloader
from nets import *
from readDataToGAN import *
from tensorboardX import SummaryWriter
import json
from utils import *
from utils import *
# from Parallel_main import parse_args

class WGAN_GP(object):
    def __init__(self, data_loader, valdata, dataset_type, train_type):
        # parameters
        args = parse_args()

        # def __init__(self, args,data_loader,valdata):
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
        self.lambda_ = 10
        self.n_critic = 5   # the number of iterations of the critic per generator iteration
        self.train_hist = {}

        self.dataset = dataset_type
        self.model_name = self.__class__.__name__ + '_' + train_type

        # self.y_real_, self.y_fake_ = torch.ones(self.batch_size, 1), torch.zeros(self.batch_size, 1)
        self.y_real_, self.y_fake_ = torch.zeros(self.batch_size, 1), torch.ones(self.batch_size, 1)

        if self.gpu_mode:
            self.y_real_, self.y_fake_ = self.y_real_.cuda(), self.y_fake_.cuda()

        # print('run at WGAN_GP')
        # load dataset
        # self.data_loader = dataloader(self.dataset, self.input_size, self.batch_size)
        # data = self.data_loader.__iter__().__next__()[0]

        # self.data_loader = testToGAN(self.dataset, 'train')

        self.data_loader = data_loader
        self.valdata = valdata

        # self.dataset = 'StepLR'
        data = next(iter(self.data_loader ))[0]

        # networks init
        self.G = generator(input_dim=self.z_dim, output_dim=data.shape[1], input_size=self.input_size)
        self.D = discriminator(input_dim=data.shape[1], output_dim=1, input_size=self.input_size)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))
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
        self.writer = SummaryWriter()#log_dir=log_dir,
        self.X = 0

        # fixed noise
        self.sample_z_ = torch.rand((self.batch_size, self.z_dim))
        if self.gpu_mode:
            self.sample_z_ = self.sample_z_.cuda()
        print('Training {},started at {}'.format(self.model_name, time.asctime(time.localtime(time.time()))),end=',')


    def train(self):
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        self.D.train()
        # print('WGAN_GP training start!!,epoch:{},module stored at:{}'.format(self.epoch,self.dataset))
        print('training start!!,data set:{},epoch:{}'.format(self.dataset,self.epoch))
        start_time = time.time()
        # url = os.path.join(self.save_dir, self.dataset, self.model_name)

        # 等间隔调整学习率 StepLR
        # schedule_G = torch.optim.lr_scheduler.StepLR(self.G_optimizer, 20, gamma=0.1, last_epoch=-1)
        # schedule_D = torch.optim.lr_scheduler.StepLR(self.D_optimizer, 30, gamma=0.1, last_epoch=-1)

        # 余弦退火调整学习率 CosineAnnealingLR
        # schedule_D=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=0, last_epoch=-1)

        # 自适应调整学习率 ReduceLROnPlateau
        """当验证集的 loss 不再下降时，进行学习率调整；或者监测验证集的 accuracy，
        当accuracy 不再上升时，则调整学习率。"""
        # torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False,
        # threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
        # schedule_G = torch.optim.lr_scheduler.ReduceLROnPlateau(self.G_optimizer, mode='min', factor=0.1, patience=10, verbose=False,
        # threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

        for epoch in range(self.epoch):
            # if epoch == 100:
            #     self.G = torch.load(os.path.join(url,'WGAN_GP_100_G.pkl'))
            #     self.D = torch.load(os.path.join(url,'WGAN_GP_100_D.pkl'))
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
                D_real_loss = -torch.mean(D_real)

                G_ = self.G(z_)
                D_fake = self.D(G_)
                D_fake_loss = torch.mean(D_fake)

                # gradient penalty
                alpha = torch.rand((self.batch_size, 1, 1, 1))
                if self.gpu_mode:
                    alpha = alpha.cuda()

                x_hat = alpha * x_.data + (1 - alpha) * G_.data
                x_hat.requires_grad = True

                pred_hat = self.D(x_hat)
                if self.gpu_mode:
                    gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()).cuda(),
                                 create_graph=True, retain_graph=True, only_inputs=True)[0]
                else:
                    gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()),
                                     create_graph=True, retain_graph=True, only_inputs=True)[0]

                gradient_penalty = self.lambda_ * ((gradients.view(gradients.size()[0], -1).norm(2, 1) - 1) ** 2).mean()

                D_loss = D_real_loss + D_fake_loss + gradient_penalty

                D_loss.backward()
                self.D_optimizer.step()

                if ((iter+1) % self.n_critic) == 0:
                    # update G network
                    self.G_optimizer.zero_grad()

                    G_ = self.G(z_)
                    D_fake = self.D(G_)
                    G_loss = -torch.mean(D_fake)
                    self.train_hist['G_loss'].append(G_loss.item())

                    G_loss.backward()
                    self.G_optimizer.step()

                    self.train_hist['D_loss'].append(D_loss.item())

                if ((iter + 1) % 100) == 0:
                    # print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                    #       ((epoch + 1), (iter + 1), self.data_loader.dataset.__len__() // self.batch_size, D_loss.item(), G_loss.item()))

                    self.writelog("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f,G_lr: %.8f, D_lr: %.8f" %
                          ((epoch + 1), (iter + 1), self.data_loader.dataset.__len__() // self.batch_size, D_loss.item(), G_loss.item(),
                           self.G_optimizer.param_groups[0]['lr'], self.D_optimizer.param_groups[0]['lr']))

                    self.writer.add_scalar('G_loss', G_loss.item(), self.X)
                    # writer.add_scalar('G_loss', -G_loss_D, X)
                    self.writer.add_scalar('D_loss', D_loss.item(), self.X)
                    self.writer.add_scalars('cross loss', {'G_loss': D_loss.item(),'D_loss': D_loss.item()}, self.X)
                    self.X += 1

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            # with torch.no_grad():
            #     self.visualize_results((epoch+1))
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

    def visualize_results(self, epoch, fix=True):
        self.G.eval()

        if not os.path.exists(self.result_dir + '/' + self.dataset + '/' + self.model_name):
            os.makedirs(self.result_dir + '/' + self.dataset + '/' + self.model_name)

        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        if fix:
            """ fixed noise """
            samples = self.G(self.sample_z_)
        else:
            """ random noise """
            sample_z_ = torch.rand((self.batch_size, self.z_dim))
            if self.gpu_mode:
                sample_z_ = sample_z_.cuda()

            samples = self.G(sample_z_)

        if self.gpu_mode:
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)

        samples = (samples + 1) / 2
        utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name + '_epoch%03d' % epoch + '.png')

    def save(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

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