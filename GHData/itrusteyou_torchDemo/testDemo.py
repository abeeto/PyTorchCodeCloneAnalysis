import torch
print(torch.cuda_path)
print(torch.cuda.is_available())
ng = torch.cuda.device_count()
print("Devices:%d" %ng)

infos = [torch.cuda.get_device_properties(i) for i in range(ng)]
print(infos)