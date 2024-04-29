import torch
import tqdm


def optimize(loss_func,
             loss_eps=5E-3,
             loss_target=0,
             x_start=-5,
             optimizer=torch.optim.Adam,
             optimizer_kwargs=dict(),
             nsteps_max=10000,
             ):
    # define variables
    x = torch.tensor(float(x_start), requires_grad=True)
    optim = optimizer([x], **optimizer_kwargs)

    # run optimization
    xs, losses = [], []
    loss_last = 0
    for cnt in tqdm.trange(nsteps_max):
        # calc loss
        loss = loss_func(x)
        if torch.abs(loss - loss_target) < loss_eps:
            break

        # print status
        # if cnt % 100 == 0: print("cnt={}, x={}, loss={}".format(cnt, x.item(), loss.item()))
        xs.append(x.item())
        losses.append(loss.item())

        # perform optimization
        loss.backward()  # computes x.grad += dloss/dx for all parameters x
        optim.step()  # updates values x += -lr * x.grad
        optim.zero_grad()  # set x.grad = 0, for next iteration. Throws error if active?!
    return xs, losses


if __name__ == '__main__':
    loss_func = lambda x: x.pow(2)
    xs, losses = optimize(loss_func,
                          optimizer=torch.optim.RMSprop,
                          optimizer_kwargs={'lr': 0.001},
                          # loss_func=cost_func_torch.cost,
                          x_start=-5,
                          )

    len(xs)

    print("===Finished")
