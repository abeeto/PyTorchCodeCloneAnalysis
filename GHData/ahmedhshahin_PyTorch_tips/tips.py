# pretrained models
# to load to part of your network
state = model.state_dict()
state.update(partial)
model.load_state_dict(state)

# to load part of existing model to your net
net.load_state_dict(saved, strict = False)

# print no of trainiable parameters
print("No of parameters: ", sum(p.numel() for p in net.parameters() if p.requires_grad))

# freeze some layers
for name, param in net.named_parameters():
  if 'fc' not in name:
    param.requires_grad = False

    
# weighted cross-entropy
def get_weight_matrix(output, label, void_pixels):
    num_zeros = (label == 0).sum().type(torch.FloatTensor)
    num_ones  = (label == 1).sum().type(torch.FloatTensor)
    total = num_zeros + num_ones
    assert total == np.prod(label.size())

    zero_wt = num_ones / total
    one_wt = num_zeros / total
    weights = torch.zeros(label.size())
    weights[label == 0] = zero_wt
    weights[label == 1] = one_wt
    if void_pixels is not None and void_pixels.sum() != 0:
        weights[void_pixels] = 0
        if len(np.unique(weights.numpy())) != 3:
            assert np.unique(weights.numpy())[1] > 0.5
            weights[weights > 0] = 1
            # np.save('error.npy', label.cpu().numpy())
            # np.save('void.npy', void_pixels.cpu().numpy())
        assert len(np.unique(weights.numpy())) == 3 or (len(np.unique(weights.numpy())) == 2 and weights.numpy().max() == 1), np.unique(weights.numpy())
    else:
        assert (len(np.unique(weights.numpy())) == 2 or num_zeros == num_ones), np.unique(weights.numpy())
    return weights

def cross_entropy_loss_torch_version(output, label, void_pixels = None, device = "cuda:0"):
    assert (output.size() == label.size())
    wts = get_weight_matrix(output, label, void_pixels)
    crit = BCEWithLogitsLoss(weight = wts).cuda(device)
    loss = crit(output, label)
    return loss

  
# Rename a layer in state_dict
def rename_state_dict_keys(source, key_transformation, target=None):
    """
    source             -> Path to the saved state dict.
    key_transformation -> Function that accepts the old key names of the state
                          dict as the only argument and returns the new key name.
    target (optional)  -> Path at which the new state dict should be saved
                          (defaults to `source`)
    Example:
    Rename the key `layer.0.weight` `layer.1.weight` and keep the names of all
    other keys.
    ```py
    def key_transformation(old_key):
        if old_key == "layer.0.weight":
            return "layer.1.weight"
        return old_key
    rename_state_dict_keys(state_dict_path, key_transformation)
    ```
    """
    if target is None:
        target = source

    state_dict = torch.load(source)
    new_state_dict = OrderedDict()

    for key, value in state_dict.items():
        new_key = key_transformation(key)
        new_state_dict[new_key] = value

    torch.save(new_state_dict, target)
