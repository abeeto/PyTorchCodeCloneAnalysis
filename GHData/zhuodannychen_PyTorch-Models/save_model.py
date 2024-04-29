# saving a model and loading it


# save entire net
torch.save(model, 'model.pkl')

# save only parameters
torch.save(model.state_dict(), 'model.pkl')

# load entire model
model2 = torch.load('model.pkl')
prediction = model2(x)

# load only params
model3.load_state_dict(torch.load('model.pkl'))
prediction = model3(x)

# sometimes, optimizers need to be saved as well, depedning on what optimizer being used.
