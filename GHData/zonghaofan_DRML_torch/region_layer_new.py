import torch
from torch import nn

#提取浅层特征
class RegionLayer_88(nn.Module):
    def __init__(self, in_channels, grid=(8, 8)):
        super(RegionLayer_88, self).__init__()

        self.in_channels = in_channels
        self.grid = grid

        self.region_layers = dict()

        for i in range(self.grid[0]):
            for j in range(self.grid[1]):
                module_name = 'region_conv88_%d_%d' % (i, j)
                self.region_layers[module_name] = nn.Sequential(
                    nn.BatchNorm2d(self.in_channels),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels,
                              kernel_size=3, stride=1, padding=1)
                )
                self.add_module(name=module_name, module=self.region_layers[module_name])

    def forward(self, x):
        """

        :param x:   (b, c, h, w)
        :return:    (b, c, h, w)
        """

        batch_size, _, height, width = x.size()
        # print('x.size()',x.size())

        input_row_list = torch.split(x, split_size_or_sections=height//self.grid[0],dim=2)#split_size=height//self.grid[0], dim=2)
        # print(len(input_row_list))
        # print('input_row_list',input_row_list[0].shape)
        output_row_list = []

        for i, row in enumerate(input_row_list):
            input_grid_list_of_a_row = torch.split(row, split_size_or_sections=width//self.grid[1], dim=3)#split_size=width//self.grid[1], dim=3)
            # print(len(input_grid_list_of_a_row))
            # print('input_grid_list_of_a_row',input_grid_list_of_a_row[0].shape)
            output_grid_list_of_a_row = []

            for j, grid in enumerate(input_grid_list_of_a_row):
                module_name = 'region_conv88_%d_%d' % (i, j)
                grid = self.region_layers[module_name](grid.contiguous()) + grid
                output_grid_list_of_a_row.append(grid)

            output_row = torch.cat(output_grid_list_of_a_row, dim=3)
            # print('output_row',output_row.shape)
            output_row_list.append(output_row)

        output = torch.cat(output_row_list, dim=2)
        # print('output=',output.shape)

        return output
#提取语义信息
class RegionLayer_31(nn.Module):
    def __init__(self, in_channels, grid=(4, 1)):
        super(RegionLayer_31, self).__init__()

        self.in_channels = in_channels
        self.grid = grid

        self.region_layers = dict()

        #fzh changed
        for i in range(self.grid[0]-1):
            for j in range(self.grid[1]):
                module_name = 'region_conv31_%d_%d' % (i, j)
                self.region_layers[module_name] = nn.Sequential(
                    nn.BatchNorm2d(self.in_channels),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels,
                              kernel_size=3, stride=1, padding=1)
                )
                self.add_module(name=module_name, module=self.region_layers[module_name])

    def forward(self, x):
        """

        :param x:   (b, c, h, w)
        :return:    (b, c, h, w)
        """

        batch_size, _, height, width = x.size()
        # print('x.size()',x.size())

        # fzh changed
        input_row_list = torch.split(x, split_size_or_sections=[2*height//self.grid[0],height//self.grid[0],height//self.grid[0]],dim=2)#split_size_or_sections=height//self.grid[0], dim=2)
        # print(len(input_row_list))
        # print('input_row_list',input_row_list[0].shape)
        # print('input_row_list', input_row_list[1].shape)
        # print('input_row_list', input_row_list[2].shape)
        #fzh changed
        output_row_list = []

        for i, row in enumerate(input_row_list):
            input_grid_list_of_a_row = torch.split(row, split_size_or_sections=width//self.grid[1], dim=3)#split_size=width//self.grid[1], dim=3)
            # print(len(input_grid_list_of_a_row))
            # print('input_grid_list_of_a_row',input_grid_list_of_a_row[0].shape)
            output_grid_list_of_a_row = []

            for j, grid in enumerate(input_grid_list_of_a_row):
                module_name = 'region_conv31_%d_%d' % (i, j)
                grid = self.region_layers[module_name](grid.contiguous()) + grid
                output_grid_list_of_a_row.append(grid)

            output_row = torch.cat(output_grid_list_of_a_row, dim=3)
            # print('output_row',output_row.shape)
            output_row_list.append(output_row)

        output = torch.cat(output_row_list, dim=2)
        # print('output=',output.shape)

        return output


def test_split():
    import cv2
    import numpy as np
    path='./1.jpg'
    grid=[8,8]
    x=cv2.imread(path)
    print(x.shape)
    # cv2.imwrite('1.jpg',x)
    x=cv2.resize(x,(160,160))
    print(x.shape)
    x=np.expand_dims(np.transpose(x,(2,0,1)),axis=0)
    print(x.shape)
    _,_,height,width=x.shape

    x=torch.from_numpy(x)
    print(x.shape)

    input_row_list = torch.split(x, split_size_or_sections=height // grid[0],
                                 dim=2)  # split_size=height//self.grid[0], dim=2)
    print(len(input_row_list))
    print('input_row_list', input_row_list[0].shape)
    # img=input_row_list[0].numpy()
    # img=np.transpose(np.squeeze(img),(1,2,0))
    # print(img.shape)
    # cv2.imwrite('2.jpg',img)
    output_row_list = []

    for i, row in enumerate(input_row_list):
        input_grid_list_of_a_row = torch.split(row, split_size_or_sections=width // grid[1],
                                               dim=3)  # split_size=width//self.grid[1], dim=3)
        print(len(input_grid_list_of_a_row))
        print('input_grid_list_of_a_row',input_grid_list_of_a_row[0].shape)
    #     output_grid_list_of_a_row = []
#
if __name__ == '__main__':
    from torch.autograd import Variable
    x = Variable(torch.randn(2, 32, 160, 160))
    print(x.shape)
    print(x.shape[1])

    #test RegionLayer_88
  #  net = RegionLayer_88(in_channels=x.shape[1], grid=(8, 8))
  #  o=net(x)
  #  print(net)
  #  print(o.shape)

    # test_split()

    # # test RegionLayer_31 eye:nose:mouth=2:1:1
    # x = Variable(torch.randn(2, 32, 12, 12))
    # print(x.shape)
    # print(x.shape[1])
    net = RegionLayer_31(in_channels=x.shape[1], grid=(4, 1))
    o = net(x)
    print(net)
    print(o.shape)



