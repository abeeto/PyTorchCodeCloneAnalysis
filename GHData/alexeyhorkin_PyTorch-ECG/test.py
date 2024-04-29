import torch
import utils

def evaluate(model, test_dataloader, device, criterior):
    test_loss, accuracy, count = 0, 0, 0
    # treshold = args.trs_hold
    model.eval()
    with torch.no_grad():
        for batch_ind, (input, target) in enumerate(test_dataloader):
            input, target = input.to(device), target.to(device) 
            output = model(input)
            output_after_tresh = utils.apply_trashold(output, 0.1)
            tmp, length =  utils.get_accuracy(output_after_tresh, target)
            accuracy+=tmp
            count+=length
            loss = criterior(output, target)
            test_loss+=loss.item()
        print(f'Test loss is:{test_loss/len(test_dataloader):.5}')
        print(f'Accuracy is: {accuracy/count:.5}')
