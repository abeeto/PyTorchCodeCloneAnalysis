''' Either use the decorator: 
@torch.no_grad()
or use 'with' keyword'''

from impl_forward_method import *
torch.set_grad_enabled(True) # Setting to default mode
print("== File turn off grads ==")

@torch.no_grad() # This decorator will turn off grad calc
def get_all_preds(model, loader):
    all_preds = torch.tensor([])
    for batch in loader:
        images, labels = batch
        preds = model(images)
        all_preds = torch.cat((all_preds, preds), dim=0)
    return all_preds

net = MyNet()
predictions = get_all_preds(net, train_loader)
print(predictions.requires_grad) # output: False

'''OR'''
with torch.no_grad():
    predictions = get_all_preds(net, train_loader)
print(predictions.requires_grad) # output: False