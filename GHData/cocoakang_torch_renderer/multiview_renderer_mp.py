import torch
import torch.nn as nn
import torch_render
import time
import multiprocessing as mp
from torch.multiprocessing import Process, Queue, Semaphore

class Rendering_Thread(Process):
    def __init__(self,setup,process_id,name,rendering_configs,device,thread_ctr_map,use_global_frame):
        Process.__init__(self)
        print("[RENDERING PROCESS {}]forked rendering process".format(process_id))
        self.setup = setup
        self.process_id = process_id
        self.name = name
        self.rendering_configs = rendering_configs
        self.device = device
        self.need_rendering = False
        self.program_end = False
        self.use_global_frame = use_global_frame

        self.thread_ctr_map = thread_ctr_map
        self.input_queue = self.thread_ctr_map["input_queue"]
        self.output_queue = self.thread_ctr_map["output_queue"]
        self.device_sph = self.thread_ctr_map["device_sph"]
        self.data_transfer_mode = self.thread_ctr_map["data_transfer_mode"]
    
    def run(self):
        print("[RENDERING PROCESS {}] Starting".format(self.process_id))
        while True:
            input_data = self.input_queue.get()
            # print("[PROCESS {}] Got rendering data".format(self.process_id))
            self.device_sph.acquire()
            # print("[PROCESS {}] Got device".format(self.process_id))
            #data to worker's device
            # print(len(input_data))
            tmp_input_params = input_data[0].to(self.device,copy=True)
            tmp_input_positions = input_data[1].to(self.device,copy=True)
            tmp_rotate_theta = input_data[2].to(self.device,copy=True)
            tmp_shared_frame = [a.to(self.device,copy=True) for a in input_data[3]] if self.use_global_frame else None
        
            #render here
            tmp_lumi,end_points = torch_render.draw_rendering_net(
                self.setup,
                tmp_input_params,
                tmp_input_positions,
                tmp_rotate_theta,
                self.name,
                tmp_shared_frame,
                *self.rendering_configs
            )
        
            if self.data_transfer_mode == 2:
                end_points = {end_points[a_key].cpu() for a_key in end_points} 
                tmp_lumi = tmp_lumi.cpu()

            self.output_queue.put([self.process_id,tmp_lumi,end_points]) 
            self.device_sph.release()

            del input_data

class Multiview_Renderer(nn.Module):
    def __init__(self,args,data_transfer_mode=1,max_process_live_per_gpu=7):
        '''
        data_transfer_mode: 
            0: all data are copied directly between GPUs. Memory leak may caused. Max rendering process num are bounded.
            1: Sending data pass CPU memory. Memory leak may caused. Max rendering process num are bounded.
            2: Both sending and returned data will pass CPU memory.  
        max_process_live_per_gpu:
            how many live process can run in one gpu
        '''
        super(Multiview_Renderer,self).__init__()
    
        ########################################
        ##parse configuration                ###
        ########################################
        self.available_devices = args["available_devices"]
        self.available_devices_num = len(self.available_devices)
        self.rendering_view_num = args["rendering_view_num"]
        self.setup = args["setup"]
        self.use_global_frame = True if (len(args["renderer_configs"]) > 0) else False
        self.renderer_name_base = args["renderer_name_base"]
        self.renderer_configs = args["renderer_configs"]#rotate point rotate normal etc.
        
        self.data_transfer_mode = data_transfer_mode
        if self.data_transfer_mode == 0:
            assert self.rendering_view_num < 7,"Data transfer mode:{}, sample view num should be less than 7, now:{}".format(self.data_transfer_mode,self.rendering_view_num)
        elif self.data_transfer_mode == 1:
            process_per_gpu = self.rendering_view_num//self.available_devices_num
            assert process_per_gpu < 15,"Data transfer mode:{}, too many rendering process on same gpu, now:{}".format(self.data_transfer_mode,process_per_gpu)        

        #######################################
        ## construct renderer               ###
        #######################################
        self.device_sph_list = []
        for which_device in range(self.available_devices_num):
            self.device_sph_list.append(Semaphore(max_process_live_per_gpu))

        self.renderer_list = []
        self.input_queue_list = []
        self.output_queue = Queue(self.rendering_view_num)
        for which_renderer in range(self.rendering_view_num):
            print("[MULTIVIEW RENDERER] create renderer:{}".format(which_renderer))
            tmp_input_queue = Queue()
            self.input_queue_list.append(tmp_input_queue)
            
            cur_device_id = which_renderer%self.available_devices_num
            cur_device = self.available_devices[cur_device_id]
            cur_semaphore = self.device_sph_list[cur_device_id]

            thread_ctr_map = {
                "input_queue":tmp_input_queue,
                "output_queue":self.output_queue,
                "device_sph":cur_semaphore,
                "data_transfer_mode":self.data_transfer_mode
            }


            tmp_renderer = Rendering_Thread(
                self.setup,
                which_renderer,
                self.renderer_name_base+"_{}".format(which_renderer),
                self.renderer_configs,
                cur_device,
                thread_ctr_map,
                self.use_global_frame
            )
            tmp_renderer.daemon = True
            tmp_renderer.start()
            
            self.renderer_list.append(tmp_renderer)
    
    def get_rendering_device_at_view(self,view_id):
        assert view_id <= self.rendering_view_num,"multiview renderer only rendering {} views:".format(self.rendering_view_num)
        return self.renderer_list[view_id].device

    def forward(self,input_params,input_positions,rotate_theta,global_frame = None,return_tensor = False,end_points_wanted_list=[]):
        '''
        input_params=(batch_size,7 or 11) torch tensor TODO:it can be a list
        input_positions=(batch_size,3) torch tensor TODO:it can be a list
        rotate_theta=(batch_size,rendering_view_num) torch tensor

        return = 
            if return_tensor = True:
                (batch, rendering_view_num, lightnum, channel_num)
            else:
                list of (batch,lightnum,channel_num) each of them on the specific gpu
        if not return_tensor:
            returned tensor will be placed on where input_params is
        else
            item of returned tensor list will be placed on where it rendered
        '''
        
        ############################################################################################################################
        ##step 0 unpack batch data
        ############################################################################################################################
        batch_size = input_params.size()[0]
        origin_device = input_params.device
        assert input_positions.size()[0] == batch_size,"input_params shape:{} input_positions shape:{}".format(input_params.size(),input_positions.size())
        all_param_dim = input_params.size()[1]
        assert all_param_dim == 11 or all_param_dim == 7,"input param dim should be 11 or 7 now:{}".format(all_param_dim)
        channel_num = 3 if all_param_dim == 11 else 1
        ############################################################################################################################
        ##step 2 rendering
        ############################################################################################################################        
        input_params = input_params.to("cpu",copy=True)
        input_positions = input_positions.to("cpu",copy=True)

        for which_view in range(self.rendering_view_num):
            tmp_rotate_theta = rotate_theta[:,[which_view]].to("cpu",copy=True)
            # print("view:{} if_use_global_frame:{}".format(which_view,self.use_global_frame))
            if self.use_global_frame:
                tmp_global_frame = [an_item.to("cpu",copy=True) for an_item in global_frame]
                self.input_queue_list[which_view].put([input_params,input_positions,tmp_rotate_theta,tmp_global_frame])
            else:
                self.input_queue_list[which_view].put([input_params,input_positions,tmp_rotate_theta])
            
        del input_params
        del input_positions
        del tmp_rotate_theta
        
        ############################################################################################################################
        ##step 3 grab_all_rendered_result
        ############################################################################################################################

        result_tensor = torch.empty(self.rendering_view_num,batch_size,self.setup.get_light_num(),channel_num,device=origin_device) if return_tensor else [None]*self.rendering_view_num
        result_end_points_list = [None]*self.rendering_view_num
        '''
        return_tensor shape: 
            list of#(batchsize,lumilen,channel_num) or
            #(rendering_view_num,batchsize,lumilen,channel_num)
        '''

        for view_id in range(self.rendering_view_num):
            tmp_result = self.output_queue.get()
            result_tensor[tmp_result[0]] = tmp_result[1].to(origin_device,copy=True) if return_tensor else tmp_result[1].clone()
            result_end_points_list[view_id] = {a_key : tmp_result[2][a_key] for a_key in end_points_wanted_list}
        
            del tmp_result

        if return_tensor:
            result_tensor = result_tensor.permute(1,0,2,3)#(batchsize,rendering_view_num,lumilen,channel_num)
        
        return result_tensor,result_end_points_list
