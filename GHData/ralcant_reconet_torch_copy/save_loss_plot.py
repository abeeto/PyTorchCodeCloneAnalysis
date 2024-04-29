import matplotlib.pyplot as plt
import numpy as np
import os
ALL_TYPES = ["reg_loss", "style_loss", "content_loss", "temp_feature_loss", "temp_output_loss", "total_loss"]
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth
def plot(epoch, itr):
    for loss_type in ALL_TYPES:
        array_path = f"runs/loss/{loss_type}/final_reconet_epoch_{epoch}_itr_{itr}.npy"
        losses = np.load(array_path, allow_pickle=True)
        plt.plot(smooth(losses, 100), color='red')
        plt.savefig(f"runs/loss/{loss_type}/loss_plot_epoch_{epoch}_itr_{itr}.jpg")
        plt.clf()
def plot_per_epoch(max_epoch, include_previous_epochs):
    possible_epochs = range(1, max_epoch+1) if include_previous_epochs else {max_epoch}
    for loss_type in ALL_TYPES:
        all_losses = np.array([])
        for epoch in possible_epochs:
            #itr = -1

            #while True:
            #itr+=1
            path = f"runs/loss/{loss_type}/style25_epoch_{epoch}.npy"
            #if not os.path.exists(path):
            #    print(f"cannot find {path}, so stopping here")
            #    break
            print(f"Adding epoch #{epoch}")
            loss = np.load(path, allow_pickle=True)
            all_losses = np.concatenate((all_losses, loss), axis=None)
        plt.plot(smooth(all_losses, 100), color='red')
        plot_path = f"runs/loss/{loss_type}/loss_plot_upto_epoch_{epoch}.jpg"
        plt.savefig(plot_path)
        plt.clf()

plot_per_epoch(max_epoch=100, include_previous_epochs=True)
    
