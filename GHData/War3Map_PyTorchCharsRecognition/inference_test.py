# import torch
#
#
# def load_data(data_path):
#
#
# def load_model(model_type, model_path):
#     """Load model from path"""
#     model = Net()
#     model.load_state_dict(torch.load(model_path))
#     model.eval()
#     return model
#
#
# def classify_image(image_data,classifier_net, device):
#     need_resize = classifier_net.need_resize
#     with torch.no_grad():
#         for data in loader:
#             data = data.to(device)
#             if need_resize:
#                 data = data.view(-1, 28 * 28)
#
#             net_out = classifier_net(data)
#             # получаем индекс максимального значения
#             predicted = net_out.data.max(1)[1]
#
#
# input_path = input("Путь к папке с данными:")
# image_set = load_data(input_path)
# for image in image_set:
#     classify_image(image_data, classifier_net, device)