import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim
from torch.autograd import Variable


def test_network(net, trainloader):

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # Create Variables for the inputs and targets
    inputs = Variable(images)
    targets = Variable(images)

    # Clear the gradients from all Variables
    optimizer.zero_grad()

    # Forward pass, then backward pass, then update weights
    output = net.forward(inputs)
    loss = criterion(output, targets)
    loss.backward()
    optimizer.step()

    return True


def imshow(image, ax=None, title=None, normalize=True):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax


def view_recon(img, recon):
    ''' Function for displaying an image (as a PyTorch Tensor) and its
        reconstruction also a PyTorch Tensor
    '''

    fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)
    axes[0].imshow(img.numpy().squeeze())
    axes[1].imshow(recon.data.numpy().squeeze())
    for ax in axes:
        ax.axis('off')
        ax.set_adjustable('box-forced')

def view_classify(img, ps, version="M"):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(16), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(16))
    if version == "M":
        ax2.set_yticklabels(np.arange(16))
    elif version == "F":
        ax2.set_yticklabels(['ABDM_COR_T2_tse_mbh_13',
        	'AB_AXT1_fl2d_inopp_mbh_14',
        	'CALF_AXIAL_PD_25',
        	'CALF_COR_T2_tse_24',
			'CHST_COR_T2_tse_mbh_10',
			'FEET_AXIAL_PD_28',
			'FEET_COR_T2_tse_27',
			'HN_AX_FLAIR_3',
			'HN_AX_T1_VIBE_8',
			'HN_AX_tof_fl2d_4',
			'HN_COR_T2_tse_7',
			'HN_SAG_T1_mpr_ns_2',
			'PEL_AXT1_fl2d_inopp_mbh_18',
			'PEL_COR_T2_tse_mbh_17',
			'THIGH_AXIAL_PD_22',
			'THIGH_COR_T2_tse_21'], size='small');
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
    plt.savefig('test_results.pdf')
