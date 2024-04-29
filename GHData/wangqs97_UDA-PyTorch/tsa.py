import torch


def TSA(label_prob, label, loss, step, schedule, total_step, start=0.2, end=1.0):
    def get_tsa_thres(total_step=total_step, step=step, schedule=schedule):
        training_progress = float(step) / float(total_step)
        if schedule == 'linear':
            thres = training_progress
        elif schedule == 'exp':
            scale = 5
            thres = torch.exp(torch.tensor(training_progress - 1) * scale)
        elif schedule == 'log':
            scale = 5
            thres = torch.tensor(1.) - torch.exp(torch.tensor(-training_progress) * scale)
        else:
            raise RuntimeError('wrong schedule input')
        return thres * (end - start) + start

    thres = get_tsa_thres()
    y_one_hot = torch.eye(10)[label, :].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    correct_label_prob = torch.sum(y_one_hot * label_prob, dim=-1)
    loss_mask = (correct_label_prob < thres).float()
    loss *= loss_mask
    avg_loss = torch.sum(loss) / torch.sum(loss_mask) if torch.sum(loss_mask) != torch.tensor(0.) else torch.tensor(0.)
    avg_loss = avg_loss.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return thres, avg_loss


if __name__ == "__main__":
    label_prob = torch.tensor([[0.1, 0.2, 0.7, 0, 0, 0, 0, 0, 0, 0], [0.3, 0.3, 0.4, 0, 0, 0, 0, 0, 0, 0]])
    label = torch.tensor([2, 0])
    loss = torch.tensor([0.1, 0.2])
    step = 1500
    schedule = 'linear'
    total_step = 2000
    avg_loss = TSA(label_prob, label, loss, step, schedule, total_step, start=0.2, end=1.0)
    print(avg_loss)
