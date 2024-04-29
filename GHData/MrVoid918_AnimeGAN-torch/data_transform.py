from torchvision import transforms

transform = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(256),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])

inv_norm = transforms.Normalize([-1, -1, -1], [2., 2., 2.])

inv_gray_transform = transforms.Compose([inv_norm, transforms.Grayscale(num_output_channels=3)])
