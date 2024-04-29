import torch

def find_lr(model, dataloader, optimizer, low_lr=1e-8, high_lr=10, num_steps=200):
    """
    Implements: sgugger.github.io/how-do-you-find-a-good-learning-rate.html

    Instead of storing losses, implementing smoothing, figuring out when to stop, etc,
    simply log learning rate and loss to weights & biases.
    """
    curr_lr = low_lr
    optimizer.param_groups[0]['lr'] = curr_lr
    lr_multiplier = (high_lr / low_lr) ** (1 / num_steps)

    for t, batch in enumerate(tqdm(dataloader)):
        if t == num_steps:
            break

        # Calculate Loss!
        # -------------------
        # pred = model(batch)
        # loss = (pred, batch)

        loss.backward()
        _loss = loss.item()

        optimizer.step()
        optimizer.zero_grad()

        wandb.log({
            "find_lr_loss":          loss,
            "find_lr_learning_rate": curr_lr
        })

        curr_lr *= lr_multiplier
        optimizer.param_groups[0]['lr'] = curr_lr
