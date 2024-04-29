################################################ dropout #################################################
############# 训练/校验
for epoch in range(epoch_time):
	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = Variable(data), Variable(target)
		net.train()      # 将神经网络转换成测试形式, 画好图之后重置为 训练形式
		output = net(data)
		loss = loss_func(output, target)

	    optimizer.zero_grad()                 
	    loss.backward()                      
	    optimizer.step()

	    if batch_idx % 10 == 0:
	    	# print('Epoch: ', epoch, '| batch_idx: ', batch_idx, '| trainloss: ', loss)
	    	net.eval()     # 校验集上测试
	    	val_losses = []
	    	for (data, target) in val_loader:
	    		data, target = Variable(data), Variable(target)
	    		output = net(data)
	    		loss = loss_func(output, target)
	    		val_losses.append(loss.data.cpu().numpy())
	    	# print('Epoch: ', epoch, '| batch_idx: ', batch_idx, '| valloss: ', loss)

############ 训练/测试
for epoch in range(epoch_time):
	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = Variable(data), Variable(target)
		net.train()      # 将神经网络转换成测试形式, 画好图之后重置为 训练形式
		output = net(data)
		loss = loss_func(output, target)

	    optimizer.zero_grad()                 
	    loss.backward()                      
	    optimizer.step()

	    if batch_idx % 10 == 0:
	    	# print('Epoch: ', epoch, '| batch_idx: ', batch_idx, '| trainloss: ', loss)

net.eval() 
test_losses = []
for (data, target) in test_loader:
	data, target = Variable(data), Variable(target)
	output = net(data)
	loss = loss_func(output, target)
	test_losses.append(loss.data.cpu().numpy())



        







