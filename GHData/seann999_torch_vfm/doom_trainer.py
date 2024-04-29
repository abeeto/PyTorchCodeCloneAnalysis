from vfm2 import VFM
import os
import argparse
import torch
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from vizdoom import *
import random
import scipy.misc
import cv2
from tensorboard_logger import configure, log_value
import torch.nn.init
import time

parser = argparse.ArgumentParser(description='vfm')
parser.add_argument('--model', type=str, default="runs/A", metavar='G',
                    help='model path')
args = parser.parse_args()

game = DoomGame()
game.load_config("/home/sean/projects/ViZDoom/scenarios/defend_the_center.cfg")
game.set_window_visible(False)
game.init()
#actions = [[0,0,1],[0,1,0],[1,0,0]]
img_size = 210
configure(args.model, flush_secs=5)

cache = []

def preprocess_img(img):
    img = scipy.misc.imresize(img, (img_size, img_size))
    img = np.moveaxis(img, 2, 0) / 255.0
    return img

#cv2.namedWindow("activations", cv2.WINDOW_NORMAL)

def train_doom():
    action_size = 4
    batch_size = 16
    in_len = 5
    out_len = 3
    rnn_input_size = 256
    rnn_size = 256
    z_size = 32

    model = VFM(img_size=img_size, action_size=action_size, batch_size=batch_size,
                in_len=in_len, out_len=out_len, rnn_input_size=rnn_input_size, rnn_size=rnn_size, z_size=z_size).cuda()
    optimizer = optim.Adam(model.parameters(), 1e-3)#
    #optimizer = optim.RMSprop(model.parameters(), lr=1e-5, momentum=0.9, weight_decay=0.95, eps=0.01)
    iters = 0

    if os.path.isfile('{}/checkpoint.tar'.format(args.model)):
        checkpoint = torch.load('{}/checkpoint.tar'.format(args.model))
        iters = checkpoint['iters']

        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint (iters {})"
              .format(checkpoint['iters']))

        del checkpoint
    #else:
    #    torch.nn.init.normal(model.mean.weight, 0, 0.1)
    #    torch.nn.init.normal(model.logvar.weight, 0, 0.1)

    game.new_episode()
    model.train()

    past_stats = [0] * action_size

    while True:
        if iters % 100 == 0 and iters != 0:
            torch.save({
                'iters': iters,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, '{}/checkpoint.tar'.format(args.model))

        batch_encoder_images = []
        batch_target_images = []
        batch_encoder_actions = []
        batch_decoder_actions = []
        encoder_images = []
        target_images = []
        encoder_actions = []
        decoder_actions = []

        while len(cache) < 100:#len(batch_encoder_images) < batch_size:
            action_i = -1

            while not game.is_episode_finished():
                state = game.get_state()
                img = np.array(state.screen_buffer)
                img = preprocess_img(img)#.copy()

                #img = np.moveaxis(img, 0, 2).copy()
                #print(img.shape)
                #cv2.putText(img, str(action_i), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255))
                #img = np.moveaxis(img, 2, 0)

                action_i = np.random.randint(action_size)
                
                action = [0] * action_size
                action[action_i] = 1

                if action_i != 3: # 3 means idle (skip)
                    _ = game.make_action(action, 4)

                #        | | | | |
                #+-+-+-+-+-+-+-+-+
                #| | | | | 

                if len(encoder_actions) < in_len:
                    encoder_images.append(img)
                    encoder_actions.append(action)
                elif len(target_images) < out_len:
                    target_images.append(img)

                    if len(decoder_actions) < out_len:
                        decoder_actions.append(action)
                else:
                    #print(len(encoder_actions), len(decoder_actions))
                    #print(len(encoder_images), len(target_images))
                    #batch_encoder_images.append(encoder_images)
                    #batch_target_images.append(target_images)
                    #batch_encoder_actions.append(encoder_actions)
                    #batch_decoder_actions.append(decoder_actions)

                    # in_len, in_len, out_len, out_len-1
                    cache.append((encoder_images, encoder_actions, target_images, decoder_actions))

                    encoder_images, target_images = [], []
                    encoder_actions, decoder_actions = [], []

                #if len(batch_encoder_images) == batch_size:
                #    break
                if len(cache) > 100:
                    break

            if game.is_episode_finished():
                game.new_episode()
                encoder_images, target_images = [], []
                encoder_actions, decoder_actions = [], []

        del cache[np.random.randint(len(cache))]

        for _ in range(batch_size):
            a, b, c, d = random.choice(cache)
            batch_encoder_images.append(a)
            batch_encoder_actions.append(b)
            batch_target_images.append(c)
            batch_decoder_actions.append(d)

        x_np = np.array(batch_encoder_images, dtype=np.float32)
        x = Variable(torch.from_numpy(x_np)).cuda()
        
        target_x_np = np.array(batch_target_images, dtype=np.float32)
        target_x = Variable(torch.from_numpy(target_x_np)).cuda()

        acts = Variable(torch.from_numpy(np.array(batch_encoder_actions, dtype=np.float32))).cuda()
        acts2 = Variable(torch.from_numpy(np.array(batch_decoder_actions, dtype=np.float32))).cuda()

        optimizer.zero_grad()

        recon, pred, q_means, q_logvars, p_means, p_logvars = model.forward(x, acts, acts2)
        #loss, mse, kld = model.loss(target_x, recon, q_means, q_logvars, p_means, p_logvars)

        target = torch.cat([x[:, 1:, ...], target_x[:, :1, ...]], 1)
        loss, mse, kld = model.loss(target, recon, q_means, q_logvars, p_means, p_logvars)

        recon = recon.data.cpu().numpy()
        pred = pred.data.cpu().numpy()
        #print(x_np.shape)
        #print(target_x_np.shape)

        def a2str(a):
            if a == 0:
                a = "left"
            elif a == 1:
                a = "right"
            elif a == 2:
                a = "shoot"
            else:
                a = "idle"

            return a

        input_images, output_images = [], []

        for i in range(in_len):
            im = np.moveaxis(x_np[0,i], 0, 2).copy()

            if i >= 1:
                a = batch_encoder_actions[0][i-1]
                a = np.argmax(a)
                #past_stats[a] = 0.99 * past_stats[a] + 0.01 * avg_var
                text = a2str(a)
                cv2.putText(im, str(text), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))

            input_images.append(im)

        for i in range(out_len):
            im = np.moveaxis(target_x_np[0,i], 0, 2).copy()

            if i == 0:
                a = batch_encoder_actions[0][-1]
            else:
                a = batch_decoder_actions[0][i-1]

            a = np.argmax(a)
            #past_stats[a] = 0.99 * past_stats[a] + 0.01 * avg_var
            text = a2str(a)
            cv2.putText(im, str(text), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
            output_images.append(im)

        canvas = np.hstack(input_images + output_images)
        pred_imgs = []

        for i in range(pred.shape[1]):
            mat = np.moveaxis(pred[0, i], 0, 2).copy()

            avg_var = np.mean(np.exp(p_logvars[0, i, :].data.cpu().numpy()))
            cv2.putText(mat, str(avg_var), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))

            pred_imgs.append(mat)

        #canvas2 = np.hstack(
        #    [np.zeros([img_size, img_size, 3]) for i in range(in_len)] + pred_imgs)
        canvas2 = np.hstack([np.moveaxis(recon[0, i], 0, 2) for i in range(in_len)] + pred_imgs)
        #print(canvas.shape)
        #print(canvas2.shape)
        canvas = np.vstack([canvas, canvas2])
        cv2.imshow("image", canvas[:,:,[2,1,0]])

        if iters % 10 == 0:
            cv2.imwrite(args.model + "/" + str(iters) + ".png", canvas[:,:,[2,1,0]]*255.0)
        cv2.waitKey(1)

        #cv2.imwrite("{}.png".format(int(time.time())), canvas[:,:,[2,1,0]] * 255.0)

        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 40)
        optimizer.step()

        #print(past_stats)
        print("{}: {}".format(iters, loss.data[0]))

        log_value("loss", loss.data[0], iters)
        log_value("mse", mse.data[0], iters)
        log_value("kld", kld.data[0], iters)
        log_value("mean", p_means.mean().data[0], iters)
        log_value("logvar", p_logvars.mean().data[0], iters)
        iters += 1

if __name__ == "__main__":
    train_doom()