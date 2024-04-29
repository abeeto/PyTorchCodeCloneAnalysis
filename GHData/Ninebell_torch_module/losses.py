import torch


def pixel_logistic_focal_loss(target, output, alpha=2, beta=4, shape=(64,64)):

    ones = torch.ones(shape).cuda()
    zeros = torch.zeros(shape).cuda()

    ones_board = torch.where(target == 1, output, ones)
    zeros_board = torch.where(target != 1, output, zeros)

    N = torch.sum(torch.where(target == 1, target, zeros))

    epsilon = 1e-8

    ones_board = torch.pow(1-ones_board, alpha) * torch.log(ones_board+epsilon)

    zeros_board = torch.pow(1-target, beta) * torch.pow(zeros_board, alpha) * torch.log(1-zeros_board+epsilon)

    return -(ones_board+zeros_board).sum()/N
