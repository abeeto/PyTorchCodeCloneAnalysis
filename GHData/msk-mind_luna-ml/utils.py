import torch

def get_state_dict_from_git_tag(weight_tag):
    tag, weight = weight_tag.split(':')
    print (f"Getting {weight} from {tag}")
    return torch.hub.load_state_dict_from_url(
        f"https://github.com/msk-mind/luna-ml/raw/{tag}/weights/{weight}",
        map_location=torch.device('cpu')
    )
