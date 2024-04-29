import torch
import numpy as np
from reprod_log import ReprodLogger
from reprod_log import ReprodDiffHelper
import paddle

from conv_model.model import conv_paddle_model
from conv_model.model import conv_torch_model




def test_forward():


    device = "cpu"  # you can also set it as "cpu"
    torch_device = torch.device("cuda:0" if device == "gpu" else "cpu")
    paddle.set_device(device)

    """load paddle model"""
    paddle_model = conv_paddle_model().net
    paddle_model.eval()
    paddle_state_dict = paddle.load("./Data/paddle.params")
    paddle_model.set_dict(paddle_state_dict)

    """load torch model"""
    torch_model = conv_torch_model().net
    torch_model.eval()
    torch_state_dict = torch.load("./Data/torch.params")
    torch_model.load_state_dict(torch_state_dict)
    torch_model.to(torch_device)

    """load data"""
    inputs = np.load("./Data/fake_data.npy")
    # save the paddle output
    reprod_logger = ReprodLogger()
    paddle_out = paddle_model(paddle.to_tensor(inputs, dtype="float32"))
    reprod_logger.add("logits", paddle_out.cpu().detach().numpy())
    reprod_logger.save("./result/forward_paddle.npy")
     # load torch model


    torch_out = torch_model(
        torch.tensor(
            inputs, dtype=torch.float32).to('cpu'))
    # save the torch output
    reprod_logger.add("logits", torch_out.cpu().detach().numpy())
    reprod_logger.save("./result/forward_ref.npy")


if __name__ =="__main__":
    test_forward()
    diff_helper = ReprodDiffHelper()
    torch_info = diff_helper.load_info("./result/forward_ref.npy")
    paddle_info = diff_helper.load_info("./result/forward_paddle.npy")


    # compare result and produce log
    diff_helper.compare_info(torch_info, paddle_info)
    diff_helper.report(
        path="./result/log/forward_diff.log", diff_threshold=1e-5)