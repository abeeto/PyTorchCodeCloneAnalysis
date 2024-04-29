import torch
from torch.autograd import Variable
import numpy as np
from abs_models import utils as u
import foolbox
from foolbox import attacks as fa


# Evaluate results on clean data
def evalClean(model=None, test_loader=None):
    print("Evaluating single model results on clean data")
    total = 0
    correct = 0
    with torch.no_grad():
        model.eval()
        for xs, ys in test_loader:
            xs, ys = Variable(xs), Variable(ys)
            if torch.cuda.is_available():
                xs, ys = xs.cuda(), ys.cuda()
            preds1 = model(xs)
            preds_np1 = preds1.cpu().detach().numpy()
            finalPred = np.argmax(preds_np1, axis=1)
            correct += (finalPred == ys.cpu().detach().numpy()).sum()
            total += len(xs)
    acc = float(correct) / total
    print('Clean accuracy: %.2f%%' % (acc * 100))


# Evaluate results on adversarially perturbed
def evalAdvAttack(model=None, test_loader=None):
    print("Evaluating single model results on adv data")
    total = 0
    correct = 0
    model.eval()
    for xs, ys in test_loader:
        if torch.cuda.is_available():
            xs, ys = xs.cuda(), ys.cuda()
        # pytorch fast gradient method
        model.eval()
        fmodel = foolbox.models.PyTorchModel(fgsm_model,  # return logits in shape (bs, n_classes)
                                             bounds=(0., 1.),  # num_classes=10,
                                             device=u.dev())
        attack = fa.LinfProjectedGradientDescentAttack(rel_stepsize=0.01 / 0.3,
                                                       steps=100,
                                                       random_start=True, )
        xs, _, success = attack(fmodel, xs, ys, epsilons=[0.3])
        xs, ys = Variable(xs[0]), Variable(ys)
        preds1 = model(xs)
        preds_np1 = preds1.cpu().detach().numpy()
        finalPred = np.argmax(preds_np1, axis=1)
        correct += (finalPred == ys.cpu().detach().numpy()).sum()
        total += test_loader.batch_size
    acc = float(correct) / total
    print('Adv accuracy: %.2f%%' % (acc * 100))
