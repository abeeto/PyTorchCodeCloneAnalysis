import LanguageModel
import hyperparameters

hps = hyperparameters.hps
model = LanguageModel.LanguageModel()

for t in range(hps['epochs']):
    train_loss = model.train()
    test_loss = model.test()

    # print loss at intervals
    if (t+1) % hps['print_every'] == 0 or t == 0:
        print('Epoch {}, train_loss: {:.4}, test_loss: {:.4}'.format(\
            t+1, train_loss, test_loss))

    # save model at intervals
    if (t+1) % hps['save_every'] == 0:
        model.save()

    # log loss at intervals
    if (t+1) % hps['log_every'] == 0 or t == 0:
        model.log(train_loss, test_loss)
