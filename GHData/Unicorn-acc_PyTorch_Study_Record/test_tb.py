from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter("logs") # 在logs文件夹中放绘制的图

'''
            Add scalar data to summary.

            Args:
            tag (string): Data identifier # 图标的title
            scalar_value (float or string/blobname): Value to save # 保存的数值（y轴）
            global_step (int): Global step value to record # 多少步，多少步对应多少数值（x轴）
'''
# # y = 2x
# for i in range(0, 100):
#     writer.add_scalar("y=2x", scalar_value = 2*i, global_step = i) # scalar_value:y轴；global_step：x轴
#
'''
        Add image data to summary.

        Note that this requires the ``pillow`` package.

        Args:
            tag (string): Data identifier
            img_tensor (torch.Tensor, numpy.array, or string/blobname): Image data
            global_step (int): Global step value to record
            walltime (float): Optional override default walltime (time.time())
              seconds after epoch of event
            dataformats (string): Image data format specification of the form
              CHW, HWC, HW, WH, etc.
        Shape:
            img_tensor: Default is :math:`(3, H, W)`. You can use ``torchvision.utils.make_grid()`` to
            convert a batch of tensor into 3xHxW format or call ``add_images`` and let us do the job.
            Tensor with :math:`(1, H, W)`, :math:`(H, W)`, :math:`(H, W, 3)` is also suitable as long as
            corresponding ``dataformats`` argument is passed, e.g. ``CHW``, ``HWC``, ``HW``.

'''
image_path = "dataset/train/bees_image/17209602_fe5a5a746f.jpg"
img_PIL = Image.open(image_path)# PIL打开的是PIL格式，用np.array转为numpy.array格式
img_array = np.array(img_PIL) #array的shape：(H, W, 3)，要指定dataformats
writer.add_image("test", img_array, 2, dataformats="HWC")



writer.close()