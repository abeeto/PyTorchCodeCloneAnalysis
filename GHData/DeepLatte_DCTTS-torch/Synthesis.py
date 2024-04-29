import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys
from params import param
import numpy as np
from scipy.io.wavfile import write

from load_audio import textProcess, load_vocab
from data import speechDataset, collate_fn, att2img, plotAtt, plotMel
import networks_v1 as networks
import vocoder


class graph(nn.Module):
    def __init__(self, trNet):
        super(graph, self).__init__()
        self.trNet = trNet
        if self.trNet is "t2m":
            self.trainGraph = networks.t2mGraph().to(DEVICE)
            
        elif self.trNet is "SSRN":
            self.trainGraph = networks.SSRNGraph().to(DEVICE)

def dirLoad(modelNumb):
    modelPath = os.path.abspath('../DCTTS.results/model_{}'.format(modelNumb))
    logt2m = list(np.genfromtxt(os.path.join(modelPath, 't2m', 'log.csv'), delimiter=','))
    globalStep = int(logt2m[-1][0])
    t2mPATH = os.path.join(modelPath, 't2m', 'best_{}'.format(globalStep), 'bestModel_{}.pth'.format(globalStep))
    
    logSSRN = list(np.genfromtxt(os.path.join(modelPath, 'SSRN', 'log.csv'), delimiter=','))
    globalStep = int(logSSRN[0][0])
    ssrnPATH = os.path.join(modelPath, 'SSRN', 'best_{}'.format(globalStep), 'bestModel_{}.pth'.format(globalStep))

    testPATH = os.path.abspath(os.path.join(modelPath, 'synthesize'))
    wavPATH = os.path.abspath(os.path.join(testPATH, 'wav'))    
    imgPATH = os.path.abspath(os.path.join(testPATH, 'img'))
    if not os.path.exists(testPATH):
        os.mkdir(testPATH)
        os.mkdir(wavPATH)
        os.mkdir(imgPATH)

    return t2mPATH, ssrnPATH, wavPATH, imgPATH

def modelLoad(t2m, SSRN, t2mPATH, SSRNPATH):
    t2mCkpt = torch.load(t2mPATH)
    ssrnCkpt = torch.load(SSRNPATH)
    t2m.load_state_dict(t2mCkpt['model_state_dict'])
    SSRN.load_state_dict(ssrnCkpt['model_state_dict'])


def Synthesize(testLoader, idx2char, DEVICE, t2mPATH, ssrnPATH, wavPATH, imgPATH):
    # load trained model
    # sound output
    t2m = graph("t2m").trainGraph
    ssrn = graph("SSRN").trainGraph
    modelLoad(t2m, ssrn, t2mPATH, ssrnPATH) 
    
    with torch.no_grad():
        t2m.eval()
        ssrn.eval()
        globalStep = 0
        # wholeMag = torch.zeros(len(testLoader.dataset), param.max_T*param.r, param.n_mags).to(DEVICE)
        for idx, batchData in enumerate(testLoader):
            # predMel with zero values
            batchTxt, batchMel, _, textLen = batchData
            batchTxt = batchTxt.to(DEVICE)
            # # batchMel = batchMel.to(DEVICE)
            # predMel = torch.zeros(param.B,  param.max_T, param.n_mels).to(DEVICE) # (B, T/r, n_mels)
            # # At every time step, predict mel spectrogram
            # for t in range(param.max_T-1):
            #     genMel, A, _  = t2m(batchTxt, predMel) #genMel : (B, n_mels, T/r)
            #     genMel_t = genMel[:, :, t+1]  # (B, n_mels)
            #     predMel[:, t+1, :] = genMel_t

            text_lengths = torch.LongTensor(textLen).to(DEVICE)
            pos = np.zeros((batchTxt.shape[0]),dtype=int)
            predMel = torch.FloatTensor(np.zeros((len(batchTxt), 1, param.n_mels))).to(DEVICE) # (N, 1, n_mel)
            epds = torch.zeros(len(batchTxt)).to(DEVICE) - torch.ones(len(batchTxt)).to(DEVICE)
            
            K, V = t2m.TextEnc(batchTxt) # K, V : (B, d, N)
            attention = np.zeros((batchTxt.shape[0], batchTxt.shape[1], 1))

            cnt = 0
            p_idx = None
            while(1):
                v__ = None
                k__ = None
                Q = t2m.AudioEnc(torch.unsqueeze(predMel[:, -1, :], 1), True) # Q : (B, d, input_buffer)
                for v_, k_, p_ in zip(V, K, pos): # batch loop
                    p_ = np.clip(p_, 1, K.shape[2]-4)
                    if v__ is None:
                        v__ = torch.unsqueeze(v_[:, p_-1 : p_+3], 0)
                        k__ = torch.unsqueeze(k_[:, p_-1 : p_+3], 0)
                    else:
                        v__ = torch.cat([v__, torch.unsqueeze(v_[:, p_-1 : p_+3], 0)], 0)
                        k__ = torch.cat([k__, torch.unsqueeze(k_[:, p_-1 : p_+3], 0)], 0)

                R_, A, _ = t2m.AttentionNet(k__, v__, Q) 
                # r_ : (B, 2*d, input_buffer)
                # a_ : (B, 4, input_buffer)
                mel_logits = t2m.AudioDec(R_, True)
                predMel = torch.cat((predMel, mel_logits.transpose(1,2)), dim = 1)
                
                if predMel.shape[1] > 300: # magic number
                    for i, idx in enumerate(epds):
                        if epds[i] == -1:
                            epds[i] = 300
                    break

                if -1 not in epds:
                    if cnt == 0:
                        break
                    elif cnt > 0:
                        cnt = -6

                cnt += 1

                a__ = None
                for a_, p_ in zip(A, pos):
                    p_ = np.clip(p_, 1, K.shape[2]-4)
                    a_ = np.expand_dims(np.pad(a_.cpu().numpy(), ((p_-1, K.shape[2] - 3 - p_),(0,0)), mode='constant'), 0 )
                    if a__ is None:
                        a__ = a_
                    else:
                        a__ = np.concatenate([a__, a_], 0) # add attention matrix to batch
                
                attention = np.concatenate([attention, a__], 2) # concatenate along time axis
            
                p = torch.argmax(A, 1).squeeze(1)
                pos = np.add(pos, p.cpu().numpy())

                if p_idx is None:
                    p_idx = p
                else:
                    p_idx = p_idx + p

                _bound = torch.ge(p_idx, text_lengths-1) # torch.ge(input, other) : return bool input >= other
                for i, idx in enumerate(p_idx):
                    if _bound[i]:
                        if epds[i] == -1:
                            epds[i] = cnt + 6 # for spare

            t2m.AudioEnc.clear_buffer()
            t2m.AudioDec.clear_buffer()

            # SSRN
            predMag = ssrn(predMel) # Out : (B, T, n_mags)
            predMag = predMag.transpose(1, 2) # (B, n_mags, T*r)
            
            for idx, (text, mag) in enumerate(zip(batchTxt, predMag)):
                # alignment
                plotAtt(att2img(attention[idx]), text, globalStep, imgPATH)
                plotMel(predMel[idx], globalStep, imgPATH)
                # vocoder
                wav = vocoder.spectrogram2wav(mag)
                write(data=wav,filename=os.path.join(wavPATH, 'wav_{}.wav'.format(globalStep)), rate=param.sr)
                globalStep += 1
            
        
if __name__ == "__main__":
    # input text sqeuence from user or use sample sequence.
    # text processing

    modelNumb = int(sys.argv[1])
    # modelNumb = 0
    trNet = 'SSRN'

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(0)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Make Directory based on the model which has lowest loss value.
    t2mPATH, ssrnPATH, wavPATH, imgPATH = dirLoad(modelNumb)

    char2idx, idx2char = load_vocab()
    # testTxt, lenTxt = textProcess(testTxt, char2idx)
    
    # load Dataset
    testDataset = speechDataset(trNet, 2)
    testLoader = DataLoader(dataset=testDataset,
                            batch_size=param.B,
                            shuffle=False,
                            collate_fn=collate_fn,
                            drop_last=True)

    Synthesize(testLoader, idx2char, DEVICE, t2mPATH, ssrnPATH, wavPATH, imgPATH)
