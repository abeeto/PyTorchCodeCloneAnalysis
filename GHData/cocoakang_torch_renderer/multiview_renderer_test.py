import numpy as np
import torch
import torch_render
from multiview_renderer_mp import Multiview_Renderer
# from multiview_renderer_mt import Multiview_Renderer
# from multiview_renderer_naive import Multiview_Renderer
import time

if __name__ == "__main__":
    TORCH_RENDER_PATH = "./"
    standard_rendering_parameters = {
        "config_dir":TORCH_RENDER_PATH+"wallet_of_torch_renderer/blackbox20_render_configs_1x1/"
    }
    setup = torch_render.Setup_Config(standard_rendering_parameters)


    test_configs = {
        "available_devices":[torch.device("cuda:0"),torch.device("cuda:1"),torch.device("cuda:2")],
        "torch_render_path":"./",
        "rendering_view_num":24,
        "setup":setup,
        "use_global_frame":False,
        "renderer_name_base":"test_rendering",
        "renderer_configs":[],
    }

    
    renderer = Multiview_Renderer(test_configs)

    batch_size = 50
    test_params = np.random.rand(batch_size,7)
    test_positions = np.random.rand(batch_size,3)

    
    tmp_params = torch.from_numpy(test_params.astype(np.float32)).to("cuda:0")
    tmp_positions = torch.from_numpy(test_positions.astype(np.float32)).to("cuda:0")
    tmp_rot = torch.zeros(batch_size,1,device="cuda:0")
    tmp_rot_list = [tmp_rot]*test_configs["rendering_view_num"]
    
    for itr in range(200):
        if itr % 10 == 0:
            print("-------------------itr:{}".format(itr))
        if itr == 10:
            start = time.time()
        renderer(tmp_params,tmp_positions,tmp_rot_list)
        # print("waiting for next itr...")
        # time.sleep(10.0)
        # print("wait done.")
    end = time.time()
    print(end - start)