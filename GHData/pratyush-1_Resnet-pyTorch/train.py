
train_loss_1=[]
train_acc_1=[]
val_loss_1=[]
val_acc_1=[]

for epoch in range(5):
  train_losses=0
  train_accuracy=0
  for img,labels in train_loader:
    out=net(img)
    loss=criterion(out,labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    acc = ((out.argmax(dim=1) == labels).float().mean())
    train_losses+=loss/len(train_loader)
    train_accuracy+= acc/len(train_loader)
  train_loss_1.append(train_losses)
  train_acc_1.append(train_accuracy)
  print('epoch:{},train_Loss:{} ,Accuracy of the network on the  train images: {} %'.format( epoch+1,train_losses,train_accuracy))
  

  with torch.no_grad():
    num_correct=0
    validation_loss=0
    validation_acc=0
    total=0
    for img,labels in valid_loader:
      output=net(img)
      valid_loss=criterion(output,labels)
      _,predicted=torch.max(output.data,1)
      total+=labels.size(0)
      validation_loss+= valid_loss/len(valid_loader)
      acc = ((output.argmax(dim=1) == labels).float().mean())
      validation_acc+=acc/len(valid_loader)

        
    val_loss_1.append(validation_loss)
    val_acc_1.append(validation_acc)
    print('epoch:{},valid_Loss:{} ,Accuracy of the network on the  validation images: {} %'.format( epoch+1,validation_loss,validation_acc))

