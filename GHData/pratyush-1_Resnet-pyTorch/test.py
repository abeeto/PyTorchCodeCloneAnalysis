
y_true=[]
y_pred=[]
for epoch in range(5):
  with torch.no_grad():
    test_losses=0
    test_acc=0
    for i,(images, labels) in enumerate(test_loader):
        y_true.extend(labels.numpy())
        outputs = net(images)
        test_loss=criterion(outputs,labels)
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted.cpu().numpy())
        test_losses+= test_loss/len(test_loader)
        acc = ((outputs.argmax(dim=1) == labels).float().mean())
        test_acc+=acc/len(test_loader)
        del images, labels, outputs
    print('epoch:{},test_Loss:{} ,Accuracy of the network on the test images: {} %'.format( epoch+1,test_losses,test_acc))
    
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_true,y_pred)
fig, ax = plt.subplots()
ax.matshow(cm, cmap='viridis', alpha=0.3)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i,s=cm[i, j])
plt.xlabel('Predictions')
plt.ylabel('Actual values')
plt.title('Confusion Matrix')
plt.show()