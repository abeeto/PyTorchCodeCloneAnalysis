import time
import torch
from DarknetModel import *
from ptflops import get_model_complexity_info
def obtain_avg_forward_time(input, model, repeat=200):
    model.eval()
    start = time.time()
    with torch.no_grad():
        for i in range(repeat):
            output = model(input)
    avg_infer_time = (time.time() - start) / repeat

    return avg_infer_time, output
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
random_input = torch.randn((1, 3, 416, 416),requires_grad=False).to(device)
model = Darknet("./config/yolov3_prune_0.5_.cfg").to(device)
flops, params = get_model_complexity_info(model, (3,416,416), as_strings=True,print_per_layer_stat=True) #不用写batch_size大小，默认batch_size=1

model1 = Darknet("./config/yolov3_prune_0.5__prune_0.5_.cfg").to(device)
flops1, params1 = get_model_complexity_info(model1, (3,416,416), as_strings=True,print_per_layer_stat=True) #不用写batch_size大小，默认batch_size=1
print('Flops:  ' + flops)
print('Params: ' + params)
print('Flops:  ' + flops1)
print('Params: ' + params1)


# compact_forward_time, compact_output = obtain_avg_forward_time(random_input,model)
# print(compact_forward_time)